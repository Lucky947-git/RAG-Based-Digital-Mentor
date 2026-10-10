(() => {
  const $ = (s) => document.querySelector(s);
  const messagesEl = $('#messages'), input = $('#message-input'), form = $('#composer');
  const state = { session_id: null, messages: [], context: {}, busy: false, lastRecord: null };
  const now = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }).toUpperCase();
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  function persist() { try { sessionStorage.setItem('darpan_session', JSON.stringify(state)); } catch {} }
  function restore() { try { const x = JSON.parse(sessionStorage.getItem('darpan_session') || sessionStorage.getItem('viveka_session') || 'null'); if (x && Array.isArray(x.messages)) Object.assign(state, x); } catch {} }
  function toast(text) { const el=$('#toast'); el.textContent=text; el.classList.add('show'); setTimeout(()=>el.classList.remove('show'),2600); }
  function showSetup(message) { const b=$('#setup-banner'); b.textContent=message; b.classList.remove('hidden'); }
  function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
  function renderUser(text, time=now()) { const row=document.createElement('div'); row.className='user-wrap'; row.innerHTML=`<div><div class="user-card">${esc(text)}</div><div class="user-time">${esc(time)}</div></div>`; messagesEl.append(row); }
  function renderAssistant(m, time=now()) {
    const row=document.createElement('div'); row.className='assistant-wrap';
    const parts=[];
    if(m.source_supported) parts.push(`<div class="reply-part"><div class="part-label">Source-supported material</div><p>${esc(m.source_supported)}</p></div>`);
    if(m.interpretation) parts.push(`<div class="reply-part"><div class="part-label">A reflection</div><p>${esc(m.interpretation)}</p></div>`);
    if(m.follow_up) parts.push(`<div class="reply-part question"><div class="part-label">A question for you</div><p>${esc(m.follow_up)}</p></div>`);
    const passages=(m.passages||[]).map((p,i)=>{const loc=[p.volume?`Volume ${p.volume}`:null,p.chapter?`Chapter: ${p.chapter}`:null,p.page?`PDF page ${p.page}`:null].filter(Boolean).join(' · ');const unavailable=(!p.page||!p.chapter)?' Page/chapter information unavailable in source metadata.':'';return `<details class="source-details"><summary>Source ${i+1} · ${esc(p.book||'Verified passage')}</summary><p>${esc(p.text||'')}<span class="source-loc">${esc(p.book||'')}${loc?' · '+esc(loc):''}${esc(unavailable)}</span></p></details>`}).join('');
    row.innerHTML=`<div class="mentor-avatar">d<span></span></div><div><div class="assistant-card"><div class="message-meta">DARPAN <span>·</span> WITH YOU</div><p>${esc(m.acknowledgement||'Thank you for sharing that.')}</p>${parts.length?`<div class="reply-parts">${parts.join('')}</div>`:''}${passages?`<div class="source-list">${passages}</div>`:''}</div><div class="welcome-time">${esc(time)}</div></div>`;
    messagesEl.append(row);
  }
  function repaint() {
    messagesEl.innerHTML='';
    if(!state.messages.length){ messagesEl.innerHTML='<div class="welcome-wrap"><div class="mentor-avatar">d<span></span></div><div class="welcome-card"><div class="message-meta">DARPAN <span>·</span> HERE TO LISTEN</div><p>Hey, I’m glad you’re here. Whatever’s on your mind, we can take it one step at a time.</p><p class="welcome-question">What’s been weighing on you lately?</p></div><div class="welcome-time">JUST NOW</div></div>'; $('#session-label').textContent='A fresh start'; }
    else { for(const m of state.messages) m.role==='user'?renderUser(m.content,m.time):renderAssistant(m,m.time); $('#session-label').textContent=state.context.main_topic||'Finding a little clarity'; }
    scrollBottom();
  }
  async function send(text) {
    text=(text||'').trim(); if(!text||state.busy)return;
    state.busy=true; $('#send-button').disabled=true; $('#typing').classList.remove('hidden'); $('#suggestions').classList.add('hidden');
    const user={role:'user',content:text,time:now()}; state.messages.push(user); renderUser(text,user.time); input.value=''; input.style.height='auto'; scrollBottom();
    try {
      const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:state.session_id,messages:state.messages.filter(x=>x.role==='user'||x.role==='assistant').map(({role,content,acknowledgement,source_supported,interpretation,follow_up,passages})=>({role,content:content||[acknowledgement,source_supported,interpretation,follow_up].filter(Boolean).join('\n'),passages})),context:state.context,message:text})});
      const data=await response.json(); if(!response.ok) throw new Error(data.detail||'I couldn’t reach the mentor just now. Please try again.');
      state.session_id=data.session_id; state.context=data.context||state.context; const answer={...data.message,time:now()}; state.messages.push(answer); renderAssistant(answer,answer.time); persist();
    } catch(err) { state.messages.pop(); renderError(err.message); }
    finally { state.busy=false; $('#send-button').disabled=false; $('#typing').classList.add('hidden'); scrollBottom(); }
  }
  function renderError(message) { const row=document.createElement('div'); row.className='assistant-wrap'; row.innerHTML=`<div class="mentor-avatar">d</div><div class="assistant-card"><div class="message-meta">A QUICK NOTE</div><p>${esc(message)}</p></div>`; messagesEl.append(row); }
  form.addEventListener('submit',e=>{e.preventDefault();send(input.value)});
  input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,130)+'px'});
  input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit()}});
  document.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>send(b.dataset.prompt)));
  $('#clear-chat').addEventListener('click',()=>{if(state.busy)return;Object.assign(state,{session_id:null,messages:[],context:{},lastRecord:null});sessionStorage.removeItem('darpan_session');sessionStorage.removeItem('viveka_session');$('#suggestions').classList.remove('hidden');repaint();toast('A fresh page. Take your time.')});
  const sourceDialog=$('#source-dialog'); $('#sources-info').addEventListener('click',()=>sourceDialog.showModal()); $('#close-modal').addEventListener('click',()=>sourceDialog.close()); $('#modal-done').addEventListener('click',()=>sourceDialog.close());
  const summaryDialog=$('#summary-dialog'); $('#summary-close').addEventListener('click',()=>summaryDialog.close());
  $('#end-session').addEventListener('click',async()=>{
    if(!state.session_id||!state.messages.length){toast('Start a conversation first, then I can gather it for you.');return;}
    try { const r=await fetch('/api/session/end',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:state.session_id,messages:state.messages,context:state.context})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Could not save this session.');state.lastRecord=d.record;$('#summary-text').textContent='Your summary is saved as a JSON record. You can also keep a copy on your device.';const topic=d.record.main_topic||'An open conversation';$('#summary-preview').textContent=`Topic: ${topic}\n\n${d.record.problem?'What was on your mind: '+d.record.problem+'\n\n':''}Next step: ${d.record.practical_next_step||'To be discovered'}\n\nQuestions asked: ${(d.record.questions_asked||[]).length}`;persist();summaryDialog.showModal();}
    catch(err){toast(err.message)}
  });
  $('#download-summary').addEventListener('click',()=>{if(!state.lastRecord)return;const blob=new Blob([JSON.stringify(state.lastRecord,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`darpan-session-${state.session_id}.json`;a.click();URL.revokeObjectURL(a.href);toast('Your summary is ready to keep.')});
  restore(); if(state.messages.length) repaint();
  fetch('/api/status').then(r=>r.json()).then(s=>{if(!s.groq_configured)showSetup('Darpan is ready for your conversation once a Groq API key is added to .env. Your messages will stay in this browser until then.');else if(!s.rag.indexed_chunks)showSetup('Your mentor is ready. Book passages are not indexed yet, so teaching-related source claims will be withheld until the approved PDFs are added.');}).catch(()=>showSetup('Could not reach the mentor service. Start the FastAPI server, then refresh this page.'));
})();
