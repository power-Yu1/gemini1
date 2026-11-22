/**
 * Gemini 反代网关 — 部署在 Cloudflare Workers
 * 功能：
 *   1. /                    -> 内置聊天页（流式打字机）
 *   2. /v1/models           -> OpenAI 兼容模型列表（透传 Gemini 官方列表）
 *   3. /v1/chat/completions -> OpenAI 兼容对话接口（支持 stream，自动转 Gemini 格式）
 *   4. /gemini/*            -> 直通代理（原样转发 generativelanguage.googleapis.com）
 * 密钥：优先 Worker 环境变量 GEMINI_API_KEY；也支持请求头 Authorization: Bearer <gemini-key> 或 x-goog-api-key 覆盖
 */

const GOOGLE_BASE = 'https://generativelanguage.googleapis.com';

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': '*',
  'Access-Control-Max-Age': '86400',
};

function jsonResp(obj, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', ...CORS, ...extraHeaders },
  });
}

function errResp(status, message, type = 'proxy_error') {
  return jsonResp({ error: { message, type, code: status } }, status);
}

/** 从请求或环境取 Gemini key：请求里只有真 key（AIza 开头）才采信，否则回落到 Worker 自己的机密 */
function getKey(request, env) {
  const auth = request.headers.get('authorization');
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    const k = auth.slice(7).trim();
    if (k.startsWith('AIza')) return k;
  }
  const xg = (request.headers.get('x-goog-api-key') || '').trim();
  if (xg.startsWith('AIza')) return xg;
  if (env && env.GEMINI_API_KEY) return env.GEMINI_API_KEY;
  return null;
}

/** 提取 OpenAI 文本内容（兼容字符串与多段数组） */
function contentToText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    let out = '';
    for (const p of content) {
      if (!p) continue;
      if (typeof p === 'string') out += p;
      else if (typeof p.text === 'string') out += p.text;
    }
    return out;
  }
  return '';
}

/** 提取 OpenAI 用户内容为 Gemini parts（支持 data: 内联图片） */
function contentToParts(content) {
  if (typeof content === 'string') return content ? [{ text: content }] : [];
  if (!Array.isArray(content)) return [];
  const parts = [];
  for (const p of content) {
    if (!p) continue;
    if (typeof p === 'string') {
      if (p) parts.push({ text: p });
      continue;
    }
    if (typeof p.text === 'string' && p.text) {
      parts.push({ text: p.text });
      continue;
    }
    const url = p.image_url && (typeof p.image_url === 'string' ? p.image_url : p.image_url.url);
    if (url && url.startsWith('data:')) {
      const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
      if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
    }
  }
  return parts;
}

/** Gemini 不支持的 JSON Schema 字段 */
const SCHEMA_DROP = new Set(['$schema', '$id', '$ref', '$defs', 'definitions', 'additionalProperties', 'strict', 'title', 'examples', 'const', 'patternProperties', 'dependencies', 'if', 'then', 'else']);

function cleanSchema(s) {
  if (s === null || s === undefined) return { type: 'object', properties: {} };
  if (Array.isArray(s)) return s.map(cleanSchema);
  if (typeof s !== 'object') return s;
  const out = {};
  for (const key of Object.keys(s)) {
    if (SCHEMA_DROP.has(key)) continue;
    const v = s[key];
    if (key === 'properties' && v && typeof v === 'object') {
      const props = {};
      for (const pk of Object.keys(v)) props[pk] = cleanSchema(v[pk]);
      out.properties = props;
    } else if (key === 'items') {
      out.items = Array.isArray(v) ? cleanSchema(v[0] || { type: 'string' }) : cleanSchema(v);
    } else if (key === 'anyOf' || key === 'oneOf') {
      out.anyOf = (Array.isArray(v) ? v : []).map(cleanSchema);
    } else if (key === 'enum') {
      out.enum = (Array.isArray(v) ? v : []).map((x) => String(x));
    } else {
      out[key] = cleanSchema(v);
    }
  }
  if (out.properties && !out.type) out.type = 'object';
  if (!out.type && !out.anyOf && !out.properties) out.type = 'string';
  return out;
}

/** Gemini finishReason -> OpenAI finish_reason */
function mapFinish(reason) {
  switch (reason) {
    case 'STOP': return 'stop';
    case 'MAX_TOKENS': return 'length';
    case 'SAFETY':
    case 'RECITATION':
    case 'BLOCKLIST':
    case 'PROHIBITED_CONTENT': return 'content_filter';
    default: return 'stop';
  }
}

/** OpenAI messages -> Gemini contents + systemInstruction（含 tools / 函数调用） */
function toGeminiPayload(openaiBody) {
  const messages = Array.isArray(openaiBody.messages) ? openaiBody.messages : [];
  const contents = [];
  const systemParts = [];
  const nameByCallId = {};

  function pushParts(role, parts) {
    if (!parts.length) return;
    const last = contents[contents.length - 1];
    if (last && last.role === role) last.parts.push(...parts);
    else contents.push({ role, parts });
  }

  for (const m of messages) {
    if (!m) continue;
    const role = m.role;
    if (role === 'system' || role === 'developer') {
      const t = contentToText(m.content);
      if (t) systemParts.push({ text: t });
      continue;
    }
    if (role === 'tool' || role === 'function') {
      const name = m.name || nameByCallId[m.tool_call_id] || 'tool';
      const raw = contentToText(m.content);
      let respObj;
      try { respObj = JSON.parse(raw); } catch (e) { respObj = { result: raw }; }
      if (respObj === null || typeof respObj !== 'object' || Array.isArray(respObj)) respObj = { result: respObj };
      pushParts('user', [{ functionResponse: { name, response: respObj } }]);
      continue;
    }
    if (role === 'assistant') {
      const parts = [];
      const t = contentToText(m.content);
      if (t) parts.push({ text: t });
      if (Array.isArray(m.tool_calls)) {
        for (const tc of m.tool_calls) {
          const fn = (tc && tc.function) || {};
          if (!fn.name) continue;
          let args = {};
          if (typeof fn.arguments === 'string' && fn.arguments.trim()) {
            try { args = JSON.parse(fn.arguments); } catch (e) { args = { _raw: fn.arguments }; }
          } else if (fn.arguments && typeof fn.arguments === 'object') {
            args = fn.arguments;
          }
          if (tc.id) nameByCallId[tc.id] = fn.name;
          const part = { functionCall: { name: fn.name, args } };
          const sig = decodeCallId(tc.id);
          if (sig) part.thoughtSignature = sig;
          parts.push(part);
        }
      }
      pushParts('model', parts);
      continue;
    }
    const parts = contentToParts(m.content);
    pushParts('user', parts.length ? parts : [{ text: '' }]);
  }

  const payload = { contents };
  if (systemParts.length) payload.systemInstruction = { parts: systemParts };
  const gc = {};
  if (typeof openaiBody.temperature === 'number') gc.temperature = openaiBody.temperature;
  if (typeof openaiBody.top_p === 'number') gc.topP = openaiBody.top_p;
  const maxOut = openaiBody.max_tokens || openaiBody.max_completion_tokens;
  if (typeof maxOut === 'number' && maxOut > 0) gc.maxOutputTokens = maxOut;
  if (typeof openaiBody.stop === 'string') gc.stopSequences = [openaiBody.stop];
  if (Array.isArray(openaiBody.stop)) gc.stopSequences = openaiBody.stop.filter((x) => typeof x === 'string');

  if (Array.isArray(openaiBody.tools) && openaiBody.tools.length) {
    const decls = [];
    for (const t of openaiBody.tools) {
      const fn = t && (t.function || (t.type === 'function' ? t : null));
      if (!fn || !fn.name) continue;
      const params = cleanSchema(fn.parameters || { type: 'object', properties: {} });
      decls.push({
        name: fn.name,
        description: fn.description || '',
        parameters: params && params.type ? params : { type: 'object', properties: {} },
      });
    }
    if (decls.length) payload.tools = [{ functionDeclarations: decls }];
  }
  const tcChoice = openaiBody.tool_choice;
  if (payload.tools && tcChoice !== undefined) {
    const cfg = { mode: 'AUTO' };
    if (tcChoice === 'none') cfg.mode = 'NONE';
    else if (tcChoice === 'required') cfg.mode = 'ANY';
    else if (tcChoice && typeof tcChoice === 'object' && tcChoice.function && tcChoice.function.name) {
      cfg.mode = 'ANY';
      cfg.allowedFunctionNames = [tcChoice.function.name];
    }
    payload.toolConfig = { functionCallingConfig: cfg };
  }
  if (Object.keys(gc).length) payload.generationConfig = gc;
  // 安全设置：全部放宽为 BLOCK_ONLY_HIGH，减少无谓拒答
  payload.safetySettings = [
    'HARM_CATEGORY_HARASSMENT',
    'HARM_CATEGORY_HATE_SPEECH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'HARM_CATEGORY_DANGEROUS_CONTENT',
  ].map((category) => ({ category, threshold: 'BLOCK_ONLY_HIGH' }));
  return payload;
}

function genId() {
  return 'chatcmpl-' + Math.random().toString(36).slice(2, 12) + Date.now().toString(36);
}

function genCallId() {
  return 'call_' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
}

/** 把 Gemini thoughtSignature 走私进 OpenAI 的 tool_call id（Gemini 3.x 回放必需） */
function encodeCallId(sig) {
  if (!sig) return genCallId();
  let b64 = '';
  try {
    b64 = btoa(String.fromCharCode(...new TextEncoder().encode(String(sig))));
  } catch (e) {
    b64 = String(sig);
  }
  const u = b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  return 'call_' + Math.random().toString(36).slice(2, 8) + '_ts_' + u;
}

function decodeCallId(id) {
  if (typeof id !== 'string') return null;
  const i = id.indexOf('_ts_');
  if (i === -1) return null;
  let u = id.slice(i + 4).replace(/-/g, '+').replace(/_/g, '/');
  while (u.length % 4) u += '=';
  try {
    const bin = atob(u);
    const bytes = new Uint8Array(bin.length);
    for (let k = 0; k < bin.length; k++) bytes[k] = bin.charCodeAt(k);
    return new TextDecoder().decode(bytes);
  } catch (e) {
    return null;
  }
}

/** Gemini 非流式响应 -> OpenAI 格式（含 tool_calls） */
function toOpenAIResponse(data, model) {
  const cand = (data.candidates && data.candidates[0]) || {};
  const parts = (cand.content && cand.content.parts) || [];
  let text = '';
  const toolCalls = [];
  for (const p of parts) {
    if (typeof p.text === 'string') text += p.text;
    if (p.functionCall && p.functionCall.name) {
      toolCalls.push({
        id: encodeCallId(p.thoughtSignature || p.thought_signature),
        type: 'function',
        function: {
          name: p.functionCall.name,
          arguments: JSON.stringify(p.functionCall.args || {}),
        },
      });
    }
  }
  if (!text && data.promptFeedback && data.promptFeedback.blockReason) {
    text = `[BLOCKED] ${data.promptFeedback.blockReason}`;
  }
  const message = { role: 'assistant', content: toolCalls.length ? (text || null) : text };
  if (toolCalls.length) message.tool_calls = toolCalls;
  let finish = 'stop';
  if (toolCalls.length) finish = 'tool_calls';
  else if (cand.finishReason) finish = mapFinish(cand.finishReason);
  const usage = data.usageMetadata || {};
  return {
    id: genId(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finish,
      },
    ],
    usage: {
      prompt_tokens: usage.promptTokenCount || 0,
      completion_tokens: usage.candidatesTokenCount || 0,
      total_tokens: usage.totalTokenCount || 0,
    },
  };
}

/** Gemini SSE 行 -> OpenAI chunk 字符串（含流式 tool_calls） */
function geminiChunkToOpenAI(geminiJson, model, state) {
  const cand = (geminiJson.candidates && geminiJson.candidates[0]) || {};
  const parts = (cand.content && cand.content.parts) || [];
  let text = '';
  const fnCalls = [];
  for (const p of parts) {
    if (typeof p.text === 'string') text += p.text;
    if (p.functionCall && p.functionCall.name) fnCalls.push({ fc: p.functionCall, sig: p.thoughtSignature || p.thought_signature });
  }
  const usage = geminiJson.usageMetadata || {};
  const delta = {};
  if (text) delta.content = text;
  if (fnCalls.length) {
    delta.tool_calls = fnCalls.map((item) => {
      const idx = state.toolCalls++;
      return {
        index: idx,
        id: encodeCallId(item.sig),
        type: 'function',
        function: { name: item.fc.name, arguments: JSON.stringify(item.fc.args || {}) },
      };
    });
  }
  let finish = null;
  if (cand.finishReason) {
    if (state.toolCalls > 0) finish = 'tool_calls';
    else finish = mapFinish(cand.finishReason);
  }
  const chunk = {
    id: state.id,
    object: 'chat.completion.chunk',
    created: state.created,
    model,
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finish,
      },
    ],
  };
  if (usage.totalTokenCount) {
    chunk.usage = {
      prompt_tokens: usage.promptTokenCount || 0,
      completion_tokens: usage.candidatesTokenCount || 0,
      total_tokens: usage.totalTokenCount || 0,
    };
  }
  return 'data: ' + JSON.stringify(chunk) + '\n\n';
}

/** 把 Gemini streamGenerateContent(alt=sse) 流转换成 OpenAI chunk 流 */
function sseTransform(geminiStream, model) {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = '';
  const state = { id: genId(), created: Math.floor(Date.now() / 1000), toolCalls: 0 };
  return new TransformStream({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true });
      let idx;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if (!line.startsWith('data:')) continue;
        const dataStr = line.slice(5).trim();
        if (!dataStr || dataStr === '[DONE]') continue;
        try {
          const j = JSON.parse(dataStr);
          controller.enqueue(encoder.encode(geminiChunkToOpenAI(j, model, state)));
        } catch (e) {
          // 忽略无法解析的行
        }
      }
    },
    flush(controller) {
      controller.enqueue(encoder.encode('data: [DONE]\n\n'));
    },
  });
}

/** 聊天页 HTML */
const CHAT_HTML = `<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemini 免费网关</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--muted:#8b949e;--acc:#58a6ff;--user:#1f6feb;--bot:#21262d}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,"Segoe UI",Roboto,"Noto Sans SC",sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:12px;padding:12px 20px;border-bottom:1px solid var(--border);background:var(--panel)}
header h1{font-size:16px;font-weight:600}
header .dot{width:9px;height:9px;border-radius:50%;background:#3fb950;box-shadow:0 0 8px #3fb950}
header select{background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:13px;outline:none}
header .st{font-size:12px;color:var(--muted);margin-left:auto}
#chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:78%;padding:12px 16px;border-radius:14px;line-height:1.7;font-size:14.5px;white-space:pre-wrap;word-break:break-word}
.msg.user{align-self:flex-end;background:var(--user);color:#fff;border-bottom-right-radius:4px}
.msg.bot{align-self:flex-start;background:var(--bot);border:1px solid var(--border);border-bottom-left-radius:4px}
.msg.bot.think{color:var(--muted);font-style:italic}
#inputbar{display:flex;gap:10px;padding:14px 20px;border-top:1px solid var(--border);background:var(--panel)}
#inp{flex:1;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:10px;padding:12px 14px;font-size:14px;resize:none;outline:none;font-family:inherit;max-height:160px}
#inp:focus{border-color:var(--acc)}
#send{background:var(--user);color:#fff;border:none;border-radius:10px;padding:0 22px;font-size:14px;cursor:pointer}
#send:disabled{opacity:.5;cursor:not-allowed}
#send:hover{filter:brightness(1.15)}
.welcome{align-self:center;text-align:center;color:var(--muted);margin:auto}
.welcome h2{color:var(--fg);font-size:20px;margin-bottom:8px}
</style>
</head>
<body>
<header>
  <div class="dot" id="statusDot"></div>
  <h1>Gemini 免费网关</h1>
  <select id="modelSel"></select>
  <span class="st" id="statusTxt">加载模型…</span>
</header>
<div id="chat"><div class="welcome"><h2>✦ 已就绪</h2><p>OpenAI 兼容接口：<code>/v1/chat/completions</code>（支持 stream）<br>模型列表：<code>/v1/models</code>　直通代理：<code>/gemini/*</code></p></div></div>
<div id="inputbar">
  <textarea id="inp" rows="1" placeholder="输入消息，Enter 发送，Shift+Enter 换行"></textarea>
  <button id="send">发送</button>
</div>
<script>
const chatEl=document.getElementById('chat'),inp=document.getElementById('inp'),sendBtn=document.getElementById('send');
const modelSel=document.getElementById('modelSel'),stTxt=document.getElementById('statusTxt'),dot=document.getElementById('statusDot');
let history=[],models=[];
function addMsg(t,cls){const d=document.createElement('div');d.className='msg '+cls;d.textContent=t;chatEl.appendChild(d);chatEl.scrollTop=chatEl.scrollHeight;return d}
async function loadModels(){
  try{
    const r=await fetch('/v1/models');const j=await r.json();
    models=(j.data||[]).map(m=>m.id);
    modelSel.innerHTML=models.map(m=>'<option>'+m+'</option>').join('');
    stTxt.textContent=models.length+' 个模型';dot.style.background='#3fb950';
  }catch(e){stTxt.textContent='模型列表加载失败';dot.style.background='#f85149'}
}
loadModels();
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();doSend()}});
sendBtn.addEventListener('click',doSend);
async function doSend(){
  const text=inp.value.trim();if(!text)return;
  const w=chatEl.querySelector('.welcome');if(w)w.remove();
  inp.value='';inp.style.height='auto';
  addMsg(text,'user');history.push({role:'user',content:text});
  sendBtn.disabled=true;
  const bot=addMsg('','bot');const think=addMsg('思考中…','bot think');
  let acc='';
  try{
    const resp=await fetch('/v1/chat/completions',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({model:modelSel.value,stream:true,messages:history})});
    if(!resp.ok){const e=await resp.json().catch(()=>({}));throw new Error((e.error&&e.error.message)||('HTTP '+resp.status))}
    const rd=resp.body.getReader(),dec=new TextDecoder();let buf='';
    while(true){const{done,value}=await rd.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;while((i=buf.indexOf('\\n\\n'))!==-1){
        const line=buf.slice(0,i).trim();buf=buf.slice(i+2);
        if(!line.startsWith('data:'))continue;
        const d=line.slice(5).trim();if(d==='[DONE]')continue;
        try{const j=JSON.parse(d);const t=j.choices&&j.choices[0].delta&&j.choices[0].delta.content;
          if(t){if(think.parentNode)think.remove();acc+=t;bot.textContent=acc;chatEl.scrollTop=chatEl.scrollHeight}}catch(e){}
      }
    }
    if(think.parentNode)think.remove();
    if(!acc)bot.textContent='（空回复）';
    history.push({role:'assistant',content:acc});
  }catch(e){think.remove();bot.textContent='❌ '+e.message}
  sendBtn.disabled=false;inp.focus();
}
</script>
</body>
</html>`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (request.method === 'OPTIONS') return new Response(null, { headers: CORS });

    // 1) 聊天页
    if (path === '/' || path === '/chat') {
      return new Response(CHAT_HTML, { headers: { 'content-type': 'text/html; charset=utf-8', ...CORS } });
    }

    // 健康检查
    if (path === '/health') {
      const hasKey = !!getKey(request, env);
      return jsonResp({ ok: true, has_key: hasKey, time: new Date().toISOString() });
    }

    // 2) OpenAI 兼容模型列表
    if (path === '/v1/models') {
      const key = getKey(request, env);
      if (!key) return errResp(500, 'Worker 未配置 GEMINI_API_KEY（可在 Worker 设置里添加 Secret，或请求时带 Authorization: Bearer <key>）');
      try {
        const r = await fetch(GOOGLE_BASE + '/v1beta/models?key=' + key);
        const data = await r.json();
        if (!r.ok) return errResp(r.status, (data.error && data.error.message) || 'upstream error');
        const out = (data.models || [])
          .filter((m) => (m.supportedGenerationMethods || []).includes('generateContent'))
          .map((m) => ({
            id: m.name.replace(/^models\//, ''),
            object: 'model',
            owned_by: 'google',
          }));
        return jsonResp({ object: 'list', data: out });
      } catch (e) {
        return errResp(502, 'upstream fetch failed: ' + e.message);
      }
    }

    // 3) OpenAI 兼容对话
    if (path === '/v1/chat/completions' && request.method === 'POST') {
      const key = getKey(request, env);
      if (!key) return errResp(500, 'Worker 未配置 GEMINI_API_KEY（可在 Worker 设置里添加 Secret，或请求时带 Authorization: Bearer <key>）');
      let body;
      try {
        body = await request.json();
      } catch (e) {
        return errResp(400, 'invalid JSON body');
      }
      const model = body.model || 'gemini-2.5-flash';
      const stream = !!body.stream;
      const payload = toGeminiPayload(body);
      const endpoint = stream
        ? GOOGLE_BASE + '/v1beta/models/' + encodeURIComponent(model) + ':streamGenerateContent?alt=sse&key=' + key
        : GOOGLE_BASE + '/v1beta/models/' + encodeURIComponent(model) + ':generateContent?key=' + key;
      let upstream;
      try {
        upstream = await fetch(endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } catch (e) {
        return errResp(502, 'upstream fetch failed: ' + e.message);
      }
      if (!upstream.ok) {
        let msg = 'upstream ' + upstream.status;
        try {
          const j = await upstream.json();
          msg = (j.error && j.error.message) || msg;
        } catch (e) {}
        return errResp(upstream.status, msg);
      }
      if (!stream) {
        const data = await upstream.json();
        return jsonResp(toOpenAIResponse(data, model));
      }
      return new Response(upstream.body.pipeThrough(sseTransform(upstream, model)), {
        headers: {
          'content-type': 'text/event-stream; charset=utf-8',
          'cache-control': 'no-cache',
          connection: 'keep-alive',
          ...CORS,
        },
      });
    }

    // 4) 直通代理 /gemini/<path>?<query> -> generativelanguage.googleapis.com/<path>?<query>
    if (path.startsWith('/gemini/')) {
      const key = getKey(request, env);
      let target = GOOGLE_BASE + path.slice('/gemini'.length) + url.search;
      if (key && !url.search.includes('key=') && !request.headers.get('x-goog-api-key') && !request.headers.get('authorization')) {
        target += (url.search ? '&' : '?') + 'key=' + key;
      }
      const headers = new Headers(request.headers);
      headers.delete('host');
      headers.delete('origin');
      headers.delete('referer');
      let resp;
      try {
        resp = await fetch(target, {
          method: request.method,
          headers,
          body: request.method === 'GET' || request.method === 'HEAD' ? undefined : request.body,
        });
      } catch (e) {
        return errResp(502, 'upstream fetch failed: ' + e.message);
      }
      const out = new Headers(resp.headers);
      out.set('Access-Control-Allow-Origin', '*');
      return new Response(resp.body, { status: resp.status, headers: out });
    }

    return errResp(404, 'not found. endpoints: / , /health , /v1/models , /v1/chat/completions , /gemini/*');
  },
};
