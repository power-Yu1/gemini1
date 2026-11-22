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

/** 从请求或环境取 Gemini key */
function getKey(request, env) {
  const auth = request.headers.get('authorization');
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    const k = auth.slice(7).trim();
    if (k && k !== 'none' && k !== 'null') return k;
  }
  const xg = request.headers.get('x-goog-api-key');
  if (xg) return xg.trim();
  if (env && env.GEMINI_API_KEY) return env.GEMINI_API_KEY;
  return null;
}

/** OpenAI messages -> Gemini contents + systemInstruction */
function toGeminiPayload(openaiBody) {
  const messages = Array.isArray(openaiBody.messages) ? openaiBody.messages : [];
  const contents = [];
  const systemParts = [];
  for (const m of messages) {
    if (!m || typeof m.content !== 'string') continue;
    if (m.role === 'system' || m.role === 'developer') {
      systemParts.push({ text: m.content });
    } else {
      contents.push({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      });
    }
  }
  const payload = { contents };
  if (systemParts.length) payload.systemInstruction = { parts: systemParts };
  const gc = {};
  if (typeof openaiBody.temperature === 'number') gc.temperature = openaiBody.temperature;
  if (typeof openaiBody.top_p === 'number') gc.topP = openaiBody.top_p;
  if (typeof openaiBody.max_tokens === 'number' && openaiBody.max_tokens > 0) gc.maxOutputTokens = openaiBody.max_tokens;
  if (typeof openaiBody.stop === 'string') gc.stopSequences = [openaiBody.stop];
  if (Array.isArray(openaiBody.stop)) gc.stopSequences = openaiBody.stop;
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

/** Gemini 非流式响应 -> OpenAI 格式 */
function toOpenAIResponse(data, model, wantStreamMeta) {
  const cand = (data.candidates && data.candidates[0]) || {};
  const parts = (cand.content && cand.content.parts) || [];
  let text = '';
  for (const p of parts) {
    if (typeof p.text === 'string') text += p.text;
  }
  if (!text && data.promptFeedback && data.promptFeedback.blockReason) {
    text = `[BLOCKED] ${data.promptFeedback.blockReason}`;
  }
  const usage = data.usageMetadata || {};
  return {
    id: genId(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: text },
        finish_reason: cand.finishReason === 'STOP' ? 'stop' : (cand.finishReason || 'stop').toLowerCase(),
      },
    ],
    usage: {
      prompt_tokens: usage.promptTokenCount || 0,
      completion_tokens: usage.candidatesTokenCount || 0,
      total_tokens: usage.totalTokenCount || 0,
    },
  };
}

/** Gemini SSE 行 -> OpenAI chunk JSON 字符串数组 */
function geminiChunkToOpenAI(geminiJson, model) {
  const cand = (geminiJson.candidates && geminiJson.candidates[0]) || {};
  const parts = (cand.content && cand.content.parts) || [];
  let text = '';
  for (const p of parts) {
    if (typeof p.text === 'string') text += p.text;
  }
  const usage = geminiJson.usageMetadata || {};
  const chunk = {
    id: genId(),
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        delta: text ? { content: text } : {},
        finish_reason:
          cand.finishReason && cand.finishReason !== 'STOP'
            ? cand.finishReason.toLowerCase()
            : cand.finishReason === 'STOP'
              ? 'stop'
              : null,
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
          controller.enqueue(encoder.encode(geminiChunkToOpenAI(j, model)));
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
