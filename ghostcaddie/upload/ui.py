"""Browser review UI served by the local upload service.

Everything shown comes from the API. There are no placeholder numbers: a target
with no result renders its reason, never a plausible-looking figure.
"""

PAGE = r"""<!doctype html><meta charset=utf-8>
<title>FairwayOS upload &amp; review (research)</title>
<style>
:root{--bg:#14161a;--fg:#e8e8ea;--dim:#9aa0a8;--ok:#6fd08c;--warn:#f0b451;--bad:#f07a7a;--line:#2a2e35}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0}
.sub{color:var(--dim);font-size:12px;margin-top:4px}
main{display:grid;grid-template-columns:1fr 380px;gap:16px;padding:16px;align-items:start}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{border:1px solid var(--line);border-radius:8px;padding:12px;background:#181b20}
.card h2{font-size:13px;margin:0 0 8px;color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
button{background:#232831;color:var(--fg);border:1px solid var(--line);border-radius:6px;
padding:6px 10px;cursor:pointer}button:hover{border-color:#3a414d}
button[disabled]{opacity:.45;cursor:not-allowed}
input[type=text]{background:#0f1115;color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:6px 8px;width:100%}
#stage{position:relative;width:100%;background:#000;border-radius:6px;overflow:hidden}
#frame{display:block;width:100%;object-fit:contain;max-height:62vh}
#ov{position:absolute;inset:0;pointer-events:none}
#stage.pick{cursor:crosshair}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.bar{height:6px;background:#0f1115;border-radius:3px;overflow:hidden;margin:6px 0}
.bar>i{display:block;height:100%;background:var(--ok);width:0}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:6px 4px;border-top:1px solid var(--line);vertical-align:top}
.t-observed{color:var(--ok)}.t-blocked,.t-unavailable{color:var(--bad)}.t-not_run{color:var(--warn)}
.reason{color:var(--dim);font-size:12px}
pre{white-space:pre-wrap;word-break:break-all;color:var(--dim);font-size:11px;max-height:220px;overflow:auto}
.note{color:var(--warn);font-size:12px}
</style>
<header>
  <h1>FairwayOS &mdash; upload &amp; review</h1>
  <div class=sub>Local research tool. Video stays on this machine. Nothing is uploaded anywhere,
  nothing is deployed. Results are pseudo-labels, not ground truth. No speed, carry, 3D or
  outcome probability is produced.</div>
</header>
<main>
<div>
  <div class=card>
    <h2>1 &middot; source</h2>
    <div class=row>
      <input type=file id=file accept="video/*">
      <button id=up>Upload &amp; analyse</button>
    </div>
    <div class=row style="margin-top:8px">
      <input type=text id=path placeholder="or a filename inside the import directory">
      <button id=upPath>Analyse import file</button>
    </div>
    <div id=srcInfo class=reason style="margin-top:8px"></div>
  </div>

  <div class=card style="margin-top:12px">
    <h2>2 &middot; frame &amp; reviewed seed</h2>
    <div id=stage><img id=frame alt="decoded frame"><canvas id=ov></canvas></div>
    <div class=row style="margin-top:8px">
      <button id=prev>&larr;</button>
      <input type=text id=fnum value=0 style="width:90px">
      <button id=next>&rarr;</button>
      <button id=load>Load frame</button>
      <span id=fmeta class=reason></span>
    </div>
    <div class=row style="margin-top:8px">
      <button id=pickHead>Clubhead box (2 clicks)</button>
      <button id=pickBall>Ball point (1 click)</button>
      <button id=clearPick>Clear</button>
      <button id=sendSeed disabled>Submit seed</button>
    </div>
    <div class=note style="margin-top:6px">Body runs automatically and takes no seed.
      Clubhead box and ball point are <b>assisted initialisation</b> that YOU supply: user-provided assistance, <b>not</b> human-verified or AI-verified truth. Recorded as such and bound to
      this source hash and this exact decoded frame.</div>
    <div id=pickInfo class=reason></div>
  </div>

  <div class=card style="margin-top:12px">
    <h2>3 &middot; source playback</h2>
    <video id=vid controls style="width:100%;max-height:40vh;background:#000"></video>
  </div>
</div>

<div>
  <div class=card>
    <h2>job</h2>
    <div id=jobState class=reason>no job yet</div>
    <div class=bar><i id=prog></i></div>
    <div class=row><button id=cancel disabled>Cancel</button>
      <button id=refresh disabled>Refresh</button>
      <button id=dl disabled>Download results</button>
      <button id=delMedia disabled>Delete media now</button></div>
    <div id=jobErr class=reason style="color:var(--bad)"></div>
    <div id=retention class=reason></div>
  </div>
  <div class=card style="margin-top:12px">
    <h2>targets</h2>
    <table id=targets><tr><td class=reason>run a job to see target states</td></tr></table>
  </div>
  <div class=card style="margin-top:12px">
    <h2>runtime readiness</h2>
    <div id=ready class=reason>&hellip;</div>
  </div>
  <div class=card style="margin-top:12px">
    <h2>raw result</h2><pre id=raw></pre>
  </div>
</div>
</main>
<script>
const $=s=>document.querySelector(s);
const TOKEN='__FAIRWAYOS_TOKEN__';
const H={'X-FairwayOS-Token':TOKEN};
let JOB=null, SRC=null, MODE=null, PICKS=[], POLL=null;

async function jget(u){const r=await fetch(u);if(!r.ok)throw new Error((await r.json()).error||r.status);return r.json()}

async function loadReady(){
  // /ready serves dependency_readiness, NOT the old `runtimes`/`ready` shape.
  // Reading a schema the server stopped serving made this card fail silently.
  try{const d=await jget('/ready');
    $('#ready').innerHTML=Object.entries(d.dependency_readiness).map(([k,v])=>{
      const cls=v.can_execute_now?'t-observed':'t-unavailable';
      const state=v.can_execute_now?'can execute now'
        :(v.requires_reviewed_seed&&v.interpreter_ready?'needs a reviewed seed'
          :'not ready');
      return `<div><b>${k}</b> <span class=${cls}>${state}</span>
       <span class=reason>${v.reason}</span></div>`}).join('')
      +`<div class=note style="margin-top:6px">${d.note}</div>`;
  }catch(e){$('#ready').textContent='readiness unavailable: '+e.message}
}

function renderTargets(res){
  if(!res){$('#targets').innerHTML='<tr><td class=reason>no result yet</td></tr>';return}
  $('#targets').innerHTML=Object.entries(res.targets).map(([k,t])=>{
    let extra='';
    if(t.result){extra=`<div class=reason>frames analysed ${t.result.frames_analysed},
      with observation ${t.result.frames_with_observation}, step ${t.result.sampling_step}</div>`}
    return `<tr><td><b>${k}</b> <span class=t-${t.outcome}>${t.outcome}</span>
      <div class=reason>${t.reason}</div>${extra}</td></tr>`}).join('')
   +`<tr><td><b>three target success</b>
      <span class="${res.three_target_success?'t-observed':'t-unavailable'}">${res.three_target_success}</span>
      <div class=reason>${res.note||''}</div></td></tr>`;
}

async function poll(){
  if(!JOB)return;
  try{
    const j=await jget('/jobs/'+JOB);
    $('#jobState').textContent=`${j.id} — ${j.state}`;
    $('#prog').style.width=Math.round((j.progress||0)*100)+'%';
    $('#jobErr').textContent=j.error||'';
    // Rendered from the job's own state, not from the click handler, so the
    // next poll re-renders it instead of wiping it.
    const mr=j.media_retention||(j.result&&j.result.media_retention);
    $('#retention').textContent=mr
      ?`media: ${mr.retained?'retained for review':'not available'} — ${mr.reason}`:'';
    renderTargets(j.result);
    $('#raw').textContent=j.result?JSON.stringify(j.result,null,1):'';
    $('#dl').disabled=!j.result;
    $('#delMedia').disabled=!JOB;
    if(['done','failed','cancelled'].includes(j.state)){clearInterval(POLL);POLL=null;$('#cancel').disabled=true}
  }catch(e){$('#jobErr').textContent=e.message}
}

async function startJob(body,isForm){
  const r=await fetch('/jobs',{method:'POST',headers:H,body:isForm?body:new URLSearchParams(body)});
  const j=await r.json();
  if(!r.ok){$('#jobErr').textContent=j.error||('HTTP '+r.status);return}
  JOB=j.id;$('#cancel').disabled=false;$('#refresh').disabled=false;$('#jobErr').textContent='';
  const meta=await jget('/jobs/'+JOB);
  SRC=null;
  $('#vid').src='/video?job='+JOB;
  await loadFrame(0);
  if(POLL)clearInterval(POLL);POLL=setInterval(poll,900);poll();
}

$('#up').onclick=()=>{const f=$('#file').files[0];if(!f){$('#jobErr').textContent='choose a file';return}
  const fd=new FormData();fd.append('video',f);startJob(fd,true)};
$('#upPath').onclick=()=>{const p=$('#path').value.trim();if(!p){$('#jobErr').textContent='enter a filename from the import directory';return}
  startJob({name:p},false)};
$('#cancel').onclick=async()=>{await fetch('/jobs/'+JOB+'/cancel',{method:'POST',headers:H});poll()};
$('#refresh').onclick=poll;
// Results download needs no new endpoint: the job JSON is already fetched.
$('#dl').onclick=async()=>{
  const j=await jget('/jobs/'+JOB);
  const b=new Blob([JSON.stringify(j.result,null,1)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(b); a.download=`fairwayos_${JOB}_results.json`;
  document.body.appendChild(a); a.click(); a.remove();
};
$('#delMedia').onclick=async()=>{
  const r=await fetch('/jobs/'+JOB+'/media/delete',{method:'POST',headers:H});
  const d=await r.json();
  if(!r.ok){$('#jobErr').textContent=d.error||'delete failed';return}
  $('#jobErr').textContent='';
  await poll();          // the retention line is rendered from job state
};

async function loadFrame(n){
  if(!JOB)return;
  const r=await fetch(`/frame?job=${JOB}&n=${n}`);
  if(!r.ok){$('#fmeta').textContent=(await r.json()).error;return}
  SRC={sha:r.headers.get('X-Source-Sha256'),
       nw:+r.headers.get('X-Native-Width'), nh:+r.headers.get('X-Native-Height'),
       req:+r.headers.get('X-Requested-Frame'), got:+r.headers.get('X-Decoded-Frame')};
  const b=await r.blob();
  $('#frame').src=URL.createObjectURL(b);
  $('#fnum').value=SRC.req;
  $('#fmeta').innerHTML=`native ${SRC.nw}&times;${SRC.nh} · requested f${SRC.req} · decoded f${SRC.got}`
    +(SRC.req!==SRC.got?' <span class=t-unavailable>(decoder returned a different frame — seeding is disabled for this frame)</span>':'')
    +`<br>source ${SRC.sha.slice(0,16)}…`;
  PICKS=[];draw();
}
$('#load').onclick=()=>loadFrame(parseInt($('#fnum').value||'0',10));
$('#prev').onclick=()=>loadFrame(Math.max(0,(SRC?SRC.req:0)-1));
$('#next').onclick=()=>loadFrame((SRC?SRC.req:0)+1);

// The image's CONTENT box in fractional CSS pixels.
// clientWidth/clientHeight are integer-rounded and measure the PADDING box,
// while getBoundingClientRect() is fractional and measures the BORDER box.
// Mixing the two shifted every seeded coordinate (a click intended at native
// 821,477 recorded 819.76,475.45). Origin AND scale now come from this one box.
function contentBox(img){
  const r=img.getBoundingClientRect(), cs=getComputedStyle(img);
  const n=v=>parseFloat(v)||0;
  const bl=n(cs.borderLeftWidth), bt=n(cs.borderTopWidth);
  const br=n(cs.borderRightWidth), bb=n(cs.borderBottomWidth);
  const pl=n(cs.paddingLeft), pt=n(cs.paddingTop);
  const pr=n(cs.paddingRight), pb=n(cs.paddingBottom);
  return {left:r.left+bl+pl, top:r.top+bt+pt,
          width:r.width-bl-br-pl-pr, height:r.height-bt-bb-pt-pb};
}
function tf(){const b=contentBox($('#frame'));return{native_w:SRC.nw,native_h:SRC.nh,
  display_w:b.width,display_h:b.height}}
function toNative(t,dx,dy){
  const s=Math.min(t.display_w/t.native_w,t.display_h/t.native_h);
  const cw=t.native_w*s, ch=t.native_h*s;
  const px=(t.display_w-cw)/2, py=(t.display_h-ch)/2;
  if(dx<px||dx>px+cw||dy<py||dy>py+ch)return null;   // padding: refuse, never clamp
  return [(dx-px)/s,(dy-py)/s];
}
function draw(){
  const c=$('#ov'),img=$('#frame');const cb=contentBox(img);
  c.width=cb.width;c.height=cb.height;
  const x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);
  x.strokeStyle='#4fd1ff';x.fillStyle='#4fd1ff';x.lineWidth=2;
  const t=SRC?tf():null;
  PICKS.forEach(p=>{const s=Math.min(t.display_w/t.native_w,t.display_h/t.native_h);
    const px=(t.display_w-t.native_w*s)/2, py=(t.display_h-t.native_h*s)/2;
    const dx=p[0]*s+px, dy=p[1]*s+py;
    x.beginPath();x.arc(dx,dy,4,0,7);x.fill()});
  if(MODE==='clubhead'&&PICKS.length===2){
    const s=Math.min(t.display_w/t.native_w,t.display_h/t.native_h);
    const px=(t.display_w-t.native_w*s)/2, py=(t.display_h-t.native_h*s)/2;
    const a=[PICKS[0][0]*s+px,PICKS[0][1]*s+py], b=[PICKS[1][0]*s+px,PICKS[1][1]*s+py];
    x.strokeRect(Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1]));
  }
}
$('#stage').onclick=e=>{
  if(!MODE||!SRC)return;
  const b=contentBox($('#frame'));
  const n=toNative(tf(),e.clientX-b.left,e.clientY-b.top);
  if(!n){$('#pickInfo').innerHTML='<span class=t-unavailable>click was in the letterbox padding, not on the image — ignored (never clamped)</span>';return}
  const need=MODE==='clubhead'?2:1;
  PICKS.push(n); if(PICKS.length>need)PICKS=[n];
  $('#pickInfo').textContent=`${MODE} · native ${PICKS.map(p=>`(${p[0].toFixed(1)}, ${p[1].toFixed(1)})`).join(' ')}`;
  // Never allow a seed on a frame the decoder did not actually deliver: the
  // coordinates would describe a picture the operator never saw.
  $('#sendSeed').disabled=(PICKS.length!==need)||(SRC.req!==SRC.got);
  draw();
};
$('#pickHead').onclick=()=>{MODE='clubhead';PICKS=[];$('#stage').classList.add('pick');$('#sendSeed').disabled=true;draw();$('#pickInfo').textContent='click two opposite corners of the clubhead'};
$('#pickBall').onclick=()=>{MODE='ball';PICKS=[];$('#stage').classList.add('pick');$('#sendSeed').disabled=true;draw();$('#pickInfo').textContent='click the ball centre'};
$('#clearPick').onclick=()=>{MODE=null;PICKS=[];$('#stage').classList.remove('pick');$('#sendSeed').disabled=true;draw();$('#pickInfo').textContent=''};
$('#sendSeed').onclick=async()=>{
  if(SRC.req!==SRC.got){$('#pickInfo').innerHTML=
    '<span class=t-unavailable>refusing to seed: the decoder returned f'+SRC.got+
    ' but f'+SRC.req+' was requested</span>';return}
  // frame = the frame actually DECODED and shown, not merely the one requested
  const seed={target:MODE,frame:SRC.got};
  if(MODE==='clubhead'){const[a,b]=PICKS;seed.box_xyxy=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])]}
  else seed.point_xy=PICKS[0];
  const r=await fetch('/jobs/'+JOB+'/seeds',{method:'POST',
    headers:{'Content-Type':'application/json','X-FairwayOS-Token':TOKEN},
    body:JSON.stringify({source_sha256:SRC.sha,seeds:[seed]})});
  const d=await r.json();
  $('#pickInfo').innerHTML=r.ok
    ?`<span class=t-observed>seed accepted — your assistance, bound to `
     +`${SRC.sha.slice(0,12)}… f${SRC.got}. Not human-verified, not AI-verified.</span>`
    :`<span class=t-unavailable>${d.error}</span>`;
};
window.addEventListener('resize',draw);
loadReady();
</script>
"""
