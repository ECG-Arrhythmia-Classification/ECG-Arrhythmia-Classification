export const CLASSES = [
  {code:'N', label_en:'Normal beat group', color:'#0d8068', description:'Includes normal, bundle branch block, and escape beats under the project\'s class mapping.'},
  {code:'S', label_en:'Supraventricular ectopic beat', color:'#4374c4', description:'Supraventricular premature or ectopic beats, including atrial and junctional beats.'},
  {code:'V', label_en:'Ventricular ectopic beat', color:'#d07833', description:'Includes premature ventricular contractions and ventricular escape beats.'},
  {code:'F', label_en:'Fusion beat', color:'#8d61b7', description:'Fusion of a ventricular beat and a normal beat.'},
  {code:'Q', label_en:'Unclassifiable / paced beat', color:'#637587', description:'Includes unclassifiable, paced, and paced–normal fusion beats; signal noise alone does not define this class.'},
];
export const MODELS = [{id:'cnn',name:'1D CNN',kind:'Morphology',note:'Amplitude and derivative distance'}, {id:'rnn',name:'BiLSTM',kind:'Sequence',note:'Cumulative sequence distance'}, {id:'transformer',name:'Transformer',kind:'Global context',note:'Region-weighted signal distance'}];
const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
export function validateSignal(signal){
  if(!Array.isArray(signal)||signal.length<32||signal.length>10000) throw new Error('Provide between 32 and 10,000 numeric samples from a single heartbeat.');
  if(signal.some(v=>typeof v!=='number'||!Number.isFinite(v)||Math.abs(v)>1e9)) throw new Error('Each ECG sample must be a finite number within ±1e9.');
  const m=mean(signal),sd=Math.sqrt(mean(signal.map(v=>(v-m)**2)));
  if(sd<1e-8) throw new Error('A flat signal does not contain enough information for classification.');
  return signal;
}
export function parseSignal(text,filename='signal.csv'){
  if(text.length>2*1024*1024) throw new Error('The file exceeds the 2 MB limit.');
  let values;
  const trimmed=text.replace(/^\uFEFF/,'').trim();
  if(!trimmed) throw new Error('The file is empty. Select another ECG file.');
  if(filename.toLowerCase().endsWith('.json')||trimmed.startsWith('[')||trimmed.startsWith('{')){
    let obj;try{obj=JSON.parse(trimmed)}catch{throw new Error('Invalid JSON. Use a numeric array or {"signal": [...]}.')}
    values=Array.isArray(obj)?obj:obj.signal;
  } else {
    const rows=trimmed.split(/\r?\n/).filter(r=>r.trim()).map(r=>r.trim().split(/[,;\t\s]+/));
    const number=s=>s!==''&&Number.isFinite(Number(s));
    let col=0,start=0;
    if(rows[0].some(s=>!number(s))){
      const headers=rows[0].map(s=>s.toLowerCase());
      col=headers.findIndex(s=>['signal','ecg','amplitude','value'].includes(s));
      if(col<0)throw new Error('Use a CSV column named signal, ecg, amplitude, or value. A single-column file may omit the header.');
      start=1;
    } else if(rows.length===1) {
      values=rows[0].map(Number);
    } else if(rows[0].length!==1)throw new Error('A multi-column CSV needs a header to identify the ECG column, for example time,signal.');
    if(!values){
      values=rows.slice(start).map((r,i)=>{
        if(r.length!==rows[0].length||!number(r[col]??''))throw new Error(`Invalid data on line ${i+start+1}.`);
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
  if(options.detrend){const mid=(a.length-1)/2,m=mean(a),den=a.reduce((s,_,i)=>s+(i-mid)**2,0),slope=a.reduce((s,v,i)=>s+(i-mid)*(v-m),0)/den;a=a.map((v,i)=>v-m-slope*(i-mid));steps.push('Linear detrending');}
  if(options.smooth){a=a.map((_,i)=>mean([-2,-1,0,1,2].map(d=>a[Math.max(0,Math.min(a.length-1,i+d))])));steps.push('5-sample moving average');}
  a=resample(a);steps.push('Resampling to 256 samples');
  if(options.normalize){a=zscore(a);steps.push('Z-score normalization');}
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
