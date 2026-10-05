export const CLASSES = [
  {code:'N', label_vi:'Nhóm nhịp bình thường', label_en:'Normal group', color:'#0d8068', description:'Nhóm N trong phân loại heartbeat; kết quả không kết luận tình trạng sức khỏe.'},
  {code:'S', label_vi:'Nhịp ngoại tâm thu trên thất', label_en:'Supraventricular ectopic', color:'#4374c4', description:'Nhóm S – heartbeat ngoại tâm thu trên thất.'},
  {code:'V', label_vi:'Nhịp ngoại tâm thu thất', label_en:'Ventricular ectopic', color:'#d07833', description:'Nhóm V – heartbeat ngoại tâm thu thất.'},
  {code:'F', label_vi:'Nhịp hợp nhất', label_en:'Fusion beat', color:'#8d61b7', description:'Nhóm F – heartbeat hợp nhất.'},
  {code:'Q', label_vi:'Nhịp chưa phân loại', label_en:'Unclassified beat', color:'#637587', description:'Nhóm Q là một lớp minh họa riêng; không tự động coi tín hiệu nhiễu là Q.'},
];
export const MODELS = [{id:'cnn',name:'1D CNN',kind:'Hình thái',note:'Khoảng cách biên độ + đạo hàm'}, {id:'rnn',name:'LSTM / GRU',kind:'Chuỗi',note:'Khoảng cách tích lũy theo chuỗi'}, {id:'transformer',name:'Transformer',kind:'Toàn cục',note:'Khoảng cách theo vùng tín hiệu'}];
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
export function validateSignal(signal){
  if(!Array.isArray(signal)||signal.length<32||signal.length>10000) throw new Error('Tín hiệu cần từ 32 đến 10.000 mẫu số của một heartbeat.');
  if(signal.some(v=>typeof v!=='number'||!Number.isFinite(v)||Math.abs(v)>1e9)) throw new Error('Tín hiệu chứa giá trị không hợp lệ; mỗi mẫu cần là số hữu hạn trong ±1e9.');
  const m=mean(signal),sd=Math.sqrt(mean(signal.map(v=>(v-m)**2)));
  if(sd<1e-8) throw new Error('Tín hiệu phẳng không đủ thông tin để phân loại.');
  return signal;
}
export function parseSignal(text,filename='signal.csv'){
  if(text.length>2*1024*1024) throw new Error('Tệp vượt quá giới hạn 2 MB.');
  let values;
  const trimmed=text.replace(/^\uFEFF/,'').trim();
  if(!trimmed) throw new Error('Tệp trống. Vui lòng chọn tệp ECG khác.');
  if(filename.toLowerCase().endsWith('.json')||trimmed.startsWith('[')||trimmed.startsWith('{')){
    let obj;try{obj=JSON.parse(trimmed)}catch{throw new Error('JSON không hợp lệ. Dùng một mảng số hoặc {"signal": [...]} .')}
    values=Array.isArray(obj)?obj:obj.signal;
  } else {
    const rows=trimmed.split(/\r?\n/).filter(r=>r.trim()).map(r=>r.trim().split(/[,;\t\s]+/));
    const number=s=>s!==''&&Number.isFinite(Number(s));
    let col=0,start=0;
    if(rows[0].some(s=>!number(s))){
      const headers=rows[0].map(s=>s.toLowerCase());
      col=headers.findIndex(s=>['signal','ecg','amplitude','value'].includes(s));
      if(col<0)throw new Error('CSV cần cột signal, ecg, amplitude hoặc value; tệp một cột có thể không cần tiêu đề.');
      start=1;
    } else if(rows.length===1) {
      values=rows[0].map(Number);
    } else if(rows[0].length!==1)throw new Error('CSV nhiều cột cần tiêu đề để xác định cột ECG, ví dụ time,signal.');
    if(!values){
      values=rows.slice(start).map((r,i)=>{
        if(r.length!==rows[0].length||!number(r[col]??''))throw new Error(`Dữ liệu không hợp lệ tại dòng ${i+start+1}.`);
        return Number(r[col]);
      });
    }
  }
  return validateSignal(values);
}
function resample(a,n=256){return Array.from({length:n},(_,i)=>{const p=i*(a.length-1)/(n-1),j=Math.floor(p),f=p-j;return a[j]*(1-f)+a[Math.min(j+1,a.length-1)]*f})}
function zscore(a){const m=mean(a),sd=Math.sqrt(mean(a.map(v=>(v-m)**2)));return a.map(v=>(v-m)/(sd||1));}
export function statistics(a){const m=mean(a);return {mean:m,std:Math.sqrt(mean(a.map(v=>(v-m)**2))),min:Math.min(...a),max:Math.max(...a)}}
export function preprocess(signal,options={detrend:true,smooth:true,normalize:true}){
  validateSignal(signal);let a=[...signal];const steps=[];
  if(options.detrend){const mid=(a.length-1)/2,m=mean(a),den=a.reduce((s,_,i)=>s+(i-mid)**2,0),slope=a.reduce((s,v,i)=>s+(i-mid)*(v-m),0)/den;a=a.map((v,i)=>v-m-slope*(i-mid));steps.push('Loại xu hướng tuyến tính');}
  if(options.smooth){a=a.map((_,i)=>mean([-2,-1,0,1,2].map(d=>a[Math.max(0,Math.min(a.length-1,i+d))])));steps.push('Làm mượt trung bình 5 mẫu');}
  a=resample(a);steps.push('Nội suy về 256 mẫu');
  if(options.normalize){a=zscore(a);steps.push('Chuẩn hóa Z-score');}
  validateSignal(a);
  return {signal:a,original_length:signal.length,processed_length:a.length,steps,statistics:{before:statistics(signal),after:statistics(a)}};
}
const gauss=(t,c,w)=>Math.exp(-0.5*((t-c)/w)**2);
export function syntheticBeat(code='N',n=360,noise=.025){
  let seed=42;const random=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296-.5};
  return Array.from({length:n},(_,i)=>{
    const t=i/(n-1);let v;
    if(code==='N')v=.14*gauss(t,.21,.033)-.16*gauss(t,.395,.014)+1.05*gauss(t,.43,.012)-.28*gauss(t,.465,.018)+.28*gauss(t,.7,.065);
    if(code==='S')v=.07*gauss(t,.26,.023)-.12*gauss(t,.37,.014)+.87*gauss(t,.405,.011)-.2*gauss(t,.435,.016)+.16*gauss(t,.64,.055);
    if(code==='V')v=.05*gauss(t,.18,.03)+.85*gauss(t,.43,.045)-.52*gauss(t,.515,.052)-.23*gauss(t,.73,.08);
    if(code==='F')v=.09*gauss(t,.21,.03)-.08*gauss(t,.37,.016)+.96*gauss(t,.43,.025)-.37*gauss(t,.49,.034)+.08*gauss(t,.69,.067);
    if(code==='Q')v=.23*Math.sin(2*Math.PI*t*4)*gauss(t,.49,.23)+.5*gauss(t,.56,.02)-.31*gauss(t,.61,.03);
    return v+.025*Math.sin(2*Math.PI*t)+noise*random();
  });
}
export function predictDemo(signal,options,model='cnn'){
  const start=performance.now();const processed=preprocess(signal,options),x=zscore(processed.signal);
  const scores=CLASSES.map(c=>{
    const y=zscore(preprocess(syntheticBeat(c.code,360,0),options).signal);
    let dist=mean(x.map((v,i)=>(v-y[i])**2));
    if(model==='cnn')dist+=.7*mean(x.slice(1).map((v,i)=>((v-x[i])-(y[i+1]-y[i]))**2));
    if(model==='rnn'){let sx=0,sy=0;dist=.5*dist+.5*mean(x.map((v,i)=>{sx+=v;sy+=y[i];return ((sx-sy)/8)**2}));}
    if(model==='transformer')dist=mean(x.map((v,i)=>(v-y[i])**2*(i>80&&i<155?2:.6)));
    return -2.8*dist;
  });
  const max=Math.max(...scores),ex=scores.map(s=>Math.exp(s-max)),sum=ex.reduce((a,b)=>a+b,0),probabilities=CLASSES.map((c,i)=>({...c,probability:ex[i]/sum}));
  const prediction=probabilities.reduce((a,b)=>a.probability>b.probability?a:b);
  return {model_id:model,backend:'prototype-distance',is_demo:true,prediction,confidence:prediction.probability,probabilities,inference_ms:performance.now()-start,preprocessing:processed};
}
export function downloadText(filename,text,type='application/json'){
  const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=filename;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
