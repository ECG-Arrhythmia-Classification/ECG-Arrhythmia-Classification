import React,{useState,useRef,useEffect,useMemo} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity,LayoutDashboard,GitCompareArrows,Network,FileText,Upload,Download,Play,ChevronDown,Info,FlaskConical,SlidersHorizontal,CheckCircle2,AlertCircle,LoaderCircle,Maximize2,Layers,Settings2,X} from 'lucide-react';
import {CLASSES,MODELS,syntheticBeat,preprocess,predictDemo,parseSignal,downloadText,statistics} from './signal.mjs';
import {DEFAULT_API_URL,readSetting,saveSetting,fetchApiJson,modelDescriptors,pipelineMetadata,pipelineLabel,modelStatus,testExample,predictionResult,resultExport} from './api.mjs';
import Evaluation from './Evaluation.jsx';
import JsonExportButton from './JsonExportButton.jsx';
import './styles.css';

const initialOptions={detrend:true,smooth:true,normalize:true};
const demoDescriptor={pipeline:'demo',input_length:256,is_demo:true,status:'ready'};

function Waveform({raw,processed,mode,setMode,noise,inputLength,processedPending}){
  const [zoom,setZoom]=useState(false),[hover,setHover]=useState(null),[keyboardReadout,setKeyboardReadout]=useState('');
  const showProcessed=mode==='processed'&&!processedPending;
  const values=showProcessed?processed:raw;
  useEffect(()=>{setHover(null);setKeyboardReadout('')},[mode,raw,processed,zoom]);
  let data=values;
  if(zoom){
    const center=values.indexOf(Math.max(...values)),len=Math.floor(values.length*.52),start=Math.max(0,Math.min(values.length-len,center-len/2));
    data=values.slice(Math.floor(start),Math.floor(start)+len);
  }
  const W=900,H=270,pad=28,low=Math.min(...data),high=Math.max(...data),span=Math.max(high-low,.001);
  const y=v=>H-pad-(v-low+.12*span)/(span*1.24)*(H-pad*2),x=i=>pad+i/(data.length-1)*(W-pad*2);
  const activeIndex=hover===null?null:Math.max(0,Math.min(data.length-1,hover));
  const path=data.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
  const signalColor=showProcessed?'#a5c3ff':'#62e2bc';
  return <section className="card waveform-card">
    <div className="card-head"><h2>ECG waveform</h2><div className="segmented" role="group" aria-label="Select raw or preprocessed signal">
      <button aria-pressed={mode==='raw'} className={mode==='raw'?'active':''} onClick={()=>setMode('raw')}>Raw</button>
      <button aria-pressed={mode==='processed'} className={mode==='processed'?'active':''} onClick={()=>setMode('processed')}>Preprocessed</button>
    </div></div>
    <div className="chart-meta">
      <span><i className="line-key" style={{background:signalColor}}/>{mode==='processed'&&processedPending?'Awaiting API preprocessing · raw signal':showProcessed?'Preprocessed amplitude':'Input amplitude'}</span>
      <span className="chart-readout" aria-live="off">{activeIndex===null?'Hover to inspect samples':`Sample ${activeIndex} · ${data[activeIndex].toFixed(3)}`}</span>
      <button className="icon-button" aria-label={zoom?'Zoom out':'Zoom in'} onClick={()=>{setZoom(!zoom);setHover(null)}}><Maximize2 size={15}/>{zoom?'2×':'1×'}</button>
    </div>
    <div className="chart-wrap">
      <span className="axis-label">{showProcessed?'Preprocessed amplitude':'Amplitude (input units)'}</span>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" tabIndex="0" aria-keyshortcuts="ArrowLeft ArrowRight Home End" aria-describedby="waveform-summary" role="img" aria-label={`${showProcessed?'Preprocessed':'Raw'} ECG waveform with ${data.length} samples`}
        onKeyDown={e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?data.length-1:Math.max(0,Math.min(data.length-1,(activeIndex??0)+(e.key==='ArrowRight'?1:-1)));setHover(next);setKeyboardReadout(`Sample ${next}, amplitude ${data[next].toFixed(3)}`)}}}
        onPointerLeave={()=>setHover(null)} onPointerMove={e=>{const r=e.currentTarget.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*W;setHover(Math.max(0,Math.min(data.length-1,Math.round((px-pad)/(W-pad*2)*(data.length-1)))))}}>
        <defs><pattern id="minor-grid" width="18" height="18" patternUnits="userSpaceOnUse"><path d="M 18 0 L 0 0 0 18" fill="none" stroke="#21414a" strokeWidth="1"/></pattern><pattern id="major-grid" width="90" height="90" patternUnits="userSpaceOnUse"><rect width="90" height="90" fill="url(#minor-grid)"/><path d="M 90 0 L 0 0 0 90" fill="none" stroke="#34565e" strokeWidth="1"/></pattern></defs>
        <rect width={W} height={H} fill="url(#major-grid)"/><path d={path} stroke={signalColor} strokeWidth="2.5" vectorEffect="non-scaling-stroke" fill="none" strokeLinejoin="round"/>
        {activeIndex!==null&&<g><line x1={x(activeIndex)} x2={x(activeIndex)} y1="0" y2={H} stroke="#96b4bf" strokeDasharray="4 4"/><circle cx={x(activeIndex)} cy={y(data[activeIndex])} r="5" fill="#eafaf5"/></g>}
      </svg>
      <p id="waveform-summary" className="sr-only">{`The ECG displays ${data.length} samples, with a minimum amplitude of ${low.toFixed(3)} and a maximum of ${high.toFixed(3)}. Use the left and right arrow keys to inspect samples; Home and End move to the first and last samples.`}</p>
      <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">{keyboardReadout}</span>
      <div className="x-labels"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
      <p className="x-axis-title">Relative position within the heartbeat {zoom?'· zoomed region':''}</p>
      {mode==='processed'&&processedPending&&<p className="chart-pending">Run a prediction to view the signal preprocessed by the API.</p>}
    </div>
    <div className="signal-stats">
      <div><span>Raw samples</span><strong>{raw.length}</strong></div>
      <div><span>Model input</span><strong>{inputLength||'—'} <small>samples</small></strong></div>
      <div><span>Standard deviation</span><strong>{statistics(values).std.toFixed(3)}</strong></div>
      <div><span>Simulated noise</span><strong>{noise===null?'—':`${new Intl.NumberFormat('en-US',{maximumFractionDigits:1}).format(noise*100)}%`}</strong></div>
    </div>
  </section>;
}

function Architecture(){return <div className="architecture-view">
  <div className="page-title"><div><h1>From signal to prediction</h1><p>The ECG workflow from input signal to classification results.</p></div><a className="diagram-download" href={`${import.meta.env.BASE_URL}architecture.svg`} download><Download size={16}/>Download SVG</a></div>
  <section className="card architecture-card">
    <div className="card-head"><h2>System architecture</h2><span className="badge">React + FastAPI</span></div>
    <div className="arch-flow">
      <div className="arch-node"><Upload/><b>ECG input</b><span>MIT-BIH test / CSV / JSON</span></div><div className="flow-line"/>
      <div className="arch-node"><SlidersHorizontal/><b>Preprocessing</b><span>Demo: 256 samples<br/>Checkpoint: 180 samples</span></div><div className="flow-line"/>
      <div className="arch-models">{MODELS.map(m=><div key={m.id}><Layers size={17}/><b>{m.name}</b></div>)}</div><div className="flow-line"/>
      <div className="arch-node"><CheckCircle2/><b>Prediction</b><span>Softmax · POST /predict</span></div><div className="flow-line"/>
      <div className="arch-node final-node"><Activity/><b>Analysis dashboard</b><span>Label · softmax scores · comparison</span></div>
    </div>
    <div className="arch-note"><Info size={18}/><p>The demo uses optional detrending, smoothing, and Z-score normalization, with resampling to 256 samples. Trained models use the shared fixed pipeline: 180 raw samples at 360 Hz → 0.5–40 Hz band-pass filtering → Z-score normalization, without resampling. FastAPI provides each model's configuration.</p></div>
  </section>
  <div className="two-col">
    <section className="card text-card"><h2>Input requirements</h2><ul>
      <li>One heartbeat with finite numeric values; use a single-column CSV or a column named <code>signal</code>.</li>
      <li>JSON as <code>[...]</code> or <code>{'{"signal": [...]}'}</code>.</li>
      <li>Demo: 32–10,000 input samples, resampled to 256 samples.</li>
      <li>Trained models: exactly 180 raw ECG samples at 360 Hz, without prior filtering or normalization.</li>
      <li>Class order: N, S, V, F, Q.</li>
    </ul></section>
    <section className="card text-card"><h2>API interface</h2><ul>
      <li><code>GET /health</code> · backend status.</li>
      <li><code>GET /models</code> · per-model pipeline, input length, and availability.</li>
      <li><code>POST /preprocess</code> · preprocessed signal and statistics.</li>
      <li><code>POST /predict</code> · label, softmax scores, inference time, and pipeline metadata.</li>
      <li><code>GET /examples</code> · real heartbeats from the test set.</li>
      <li><code>GET /evaluation</code> · saved evaluation and robustness results.</li>
      <li>Checkpoint errors are reported explicitly; the API does not fall back to demo mode.</li>
    </ul></section>
  </div>
</div>}

function App(){
  const [page,setPage]=useState('workspace'),[code,setCode]=useState('N'),[noise,setNoise]=useState(.025);
  const [raw,setRaw]=useState(()=>syntheticBeat('N')),[source,setSource]=useState('Synthetic sample · N'),[options,setOptions]=useState(initialOptions);
  const [model,setModel]=useState('cnn'),[chartMode,setChartMode]=useState('raw'),[results,setResults]=useState([]),[modelErrors,setModelErrors]=useState({});
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[apiOpen,setApiOpen]=useState(false);
  const [engine,setEngine]=useState(()=>readSetting('engine-v2','api')==='demo'?'demo':'api');
  const [apiUrl,setApiUrl]=useState(()=>readSetting('api-url',DEFAULT_API_URL)),[apiStatus,setApiStatus]=useState('');
  const [descriptors,setDescriptors]=useState([]),[modelsStatus,setModelsStatus]=useState('idle'),[modelsError,setModelsError]=useState('');
  const [inputMode,setInputMode]=useState('test'),[sourceInfo,setSourceInfo]=useState({is_synthetic:true}),[examples,setExamples]=useState(null),[examplesStatus,setExamplesStatus]=useState('idle'),[examplesError,setExamplesError]=useState(''),[sampleBusy,setSampleBusy]=useState(false),[testIndex,setTestIndex]=useState('0'),[testCode,setTestCode]=useState('N');
  const [evaluation,setEvaluation]=useState(null),[evaluationStatus,setEvaluationStatus]=useState('idle'),[evaluationError,setEvaluationError]=useState(''),[evaluationReload,setEvaluationReload]=useState(0),[examplesReload,setExamplesReload]=useState(0);
  const input=useRef(),modalRef=useRef(),modelsRequest=useRef(0),modelsController=useRef(null),connectionController=useRef(null);
  const inferenceGeneration=useRef(0),inferenceControllers=useRef(new Set()),sampleController=useRef(null),samplePending=useRef(false),sourceGeneration=useRef(0),loadInitialSample=useRef(true);
  const configRef=useRef('');configRef.current=`${engine}:${apiUrl}`;
  const selectedDescriptor=engine==='demo'?demoDescriptor:descriptors.find(d=>d.id===model);
  const metadataReady=engine==='demo'||modelsStatus==='ready';
  const isTeam=engine==='api'&&selectedDescriptor?.pipeline==='team';
  const comparisonHasTeam=engine==='api'&&page==='compare'&&descriptors.some(d=>d.pipeline==='team');
  const syntheticLength=isTeam||comparisonHasTeam?180:360;
  const result=results.find(r=>r.model_id===model);
  const classInfo=result?CLASSES.find(c=>c.code===result.prediction.code):null;
  const actualMetadata=pipelineMetadata(result,selectedDescriptor);
  const processed=useMemo(()=>{
    if(engine==='api')return result?.preprocessing?.signal||raw;
    try{return preprocess(raw,options).signal}catch{return raw}
  },[engine,result,raw,options]);
  const processedPending=engine==='api'&&!result?.preprocessing?.signal;
  const needsRaw180=isTeam||comparisonHasTeam;
  const apiNotReady=engine==='api'&&(modelsStatus!=='ready'||!selectedDescriptor);
  const apiOnlyDemo=engine==='api'&&modelsStatus==='ready'&&descriptors.length>0&&descriptors.every(descriptor=>descriptor.is_demo);

  const invalidate=()=>{++inferenceGeneration.current;inferenceControllers.current.forEach(controller=>controller.abort());inferenceControllers.current.clear();setBusy(false);setResults([]);setModelErrors({});setError('')};
  const cancelSample=()=>{++sourceGeneration.current;sampleController.current?.abort();samplePending.current=false;setSampleBusy(false)};

  useEffect(()=>saveSetting('engine-v2',engine),[engine]);
  useEffect(()=>saveSetting('api-url',apiUrl),[apiUrl]);
  useEffect(()=>{
    if(inputMode!=='synthetic'||!code||!metadataReady||raw.length===syntheticLength)return;
    setRaw(syntheticBeat(code,syntheticLength,noise??.025));
    setSource(`Synthetic sample · ${code}`);
    invalidate();
  },[code,noise,raw.length,syntheticLength,metadataReady,inputMode]);

  async function refreshModels(url){
    modelsController.current?.abort();
    const controller=new AbortController(),request=++modelsRequest.current;
    modelsController.current=controller;
    setModelsStatus('loading');setModelsError('');invalidate();
    const timer=setTimeout(()=>controller.abort(),8000);
    try{
      const body=await fetchApiJson(url,'/models',{signal:controller.signal});
      const next=modelDescriptors(body);
      if(!next.length)throw new Error('The API did not register any models.');
      if(next.some(d=>!['demo','team'].includes(d.pipeline)||d.input_length!==(d.pipeline==='team'?180:256)))throw new Error('The API did not provide a valid pipeline and input length for each model.');
      if(request!==modelsRequest.current||controller.signal.aborted)return null;
      setDescriptors(next);setModelsStatus('ready');
      return next;
    }catch(e){
      if(request!==modelsRequest.current)return null;
      const message=e.name==='AbortError'?'The /models endpoint did not respond within 8 seconds.':e.message;
      setDescriptors([]);setModelsStatus('error');setModelsError(message);
      throw new Error(message);
    }finally{clearTimeout(timer)}
  }

  useEffect(()=>{
    connectionController.current?.abort();
    cancelSample();invalidate();
    setApiStatus('');setDescriptors([]);setModelsError('');
    if(engine!=='api'){
      ++modelsRequest.current;modelsController.current?.abort();setModelsStatus('idle');
      return;
    }
    refreshModels(apiUrl).catch(()=>{});
    const controller=modelsController.current;
    return()=>{controller?.abort();++modelsRequest.current};
  },[engine,apiUrl]);
  useEffect(()=>()=>{modelsController.current?.abort();connectionController.current?.abort();sampleController.current?.abort();inferenceControllers.current.forEach(controller=>controller.abort())},[]);

  useEffect(()=>{
    setExamples(null);setExamplesError('');
    if(engine!=='api'){setExamplesStatus('idle');return}
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);let active=true;
    setExamplesStatus('loading');
    fetchApiJson(apiUrl,'/examples',{signal:controller.signal}).then(body=>{
      if(!Array.isArray(body.samples)||!Number.isInteger(body.total)||body.is_synthetic!==false)throw new Error('The API did not provide a valid test-sample list.');
      if(!active||controller.signal.aborted)return;
      setExamples(body);setExamplesStatus('ready');
      if(loadInitialSample.current){loadInitialSample.current=false;const first=body.samples.find(sample=>sample.code==='N')||body.samples[0];if(first)loadTestSample(first.index)}
    }).catch(e=>{if(!active)return;setExamplesStatus('error');setExamplesError(e.name==='AbortError'?'Could not load test samples. Check the API and reconnect.':e.message)}).finally(()=>clearTimeout(timer));
    return()=>{active=false;clearTimeout(timer);controller.abort()};
  },[engine,apiUrl,examplesReload]);

  useEffect(()=>{
    setEvaluation(null);setEvaluationError('');
    if(engine!=='api'){setEvaluationStatus('idle');return}
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),10000);let active=true;
    setEvaluationStatus('loading');
    fetchApiJson(apiUrl,'/evaluation',{signal:controller.signal}).then(body=>{
      if(!Array.isArray(body.models)||!body.models.length||body.models.some(model=>!Number.isFinite(model.macro_f1)||!Number.isFinite(model.accuracy)))throw new Error('The API did not provide valid evaluation metrics.');
      if(!active||controller.signal.aborted)return;
      setEvaluation(body);setEvaluationStatus('ready');
    }).catch(e=>{if(!active)return;setEvaluationStatus('error');setEvaluationError(e.name==='AbortError'?'Could not load evaluation results. Check the API and retry.':e.message)}).finally(()=>clearTimeout(timer));
    return()=>{active=false;clearTimeout(timer);controller.abort()};
  },[engine,apiUrl,evaluationReload]);

  async function loadTestSample(index){
    if(engine!=='api'){setExamplesError('Select FastAPI in Model connection to load real heartbeats.');return}
    const parsed=Number(index);
    if(String(index).trim()===''||!Number.isInteger(parsed)||parsed<0||(examples&&parsed>=examples.total)){setExamplesError(`Enter an index from 0 to ${(examples?.total||18098)-1}.`);return}
    cancelSample();invalidate();samplePending.current=true;setSampleBusy(true);setExamplesError('');
    const controller=new AbortController(),generation=sourceGeneration.current,timer=setTimeout(()=>controller.abort(),10000);sampleController.current=controller;
    try{
      const body=testExample(await fetchApiJson(apiUrl,`/examples/${parsed}`,{signal:controller.signal}));
      if(generation!==sourceGeneration.current||controller.signal.aborted)return;
      invalidate();
      setInputMode('test');setRaw(body.signal);setSource(`${body.source} · heartbeat #${body.index}`);setSourceInfo(body);setTestIndex(String(body.index));setTestCode(body.expected_class.code);setCode('');setNoise(null);setChartMode('raw');
    }catch(e){if(generation===sourceGeneration.current)setExamplesError(e.name==='AbortError'?'The test-sample request timed out after 10 seconds. Try loading it again.':e.message)}finally{clearTimeout(timer);if(generation===sourceGeneration.current){samplePending.current=false;setSampleBusy(false)}}
  }

  useEffect(()=>{
    if(!apiOpen)return;
    const before=document.activeElement,box=modalRef.current;
    const focusable=()=>Array.from(box.querySelectorAll('button,input,select,a[href]')).filter(e=>!e.disabled);
    focusable()[0]?.focus();
    const handler=e=>{
      if(e.key==='Escape')setApiOpen(false);
      if(e.key==='Tab'){
        const elements=focusable(),first=elements[0],last=elements[elements.length-1];
        if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus()}
        else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus()}
      }
    };
    document.addEventListener('keydown',handler);
    return()=>{document.removeEventListener('keydown',handler);before?.focus()};
  },[apiOpen]);

  const choose=(c,n=noise??.025)=>{
    cancelSample();loadInitialSample.current=false;setInputMode('synthetic');setSourceInfo({is_synthetic:true});setCode(c);setRaw(syntheticBeat(c,syntheticLength,n));setSource(`Synthetic sample · ${c}`);setNoise(n);invalidate();
  };
  async function upload(file){
    if(!file)return;
    try{
      if(file.size>2*1024*1024)throw new Error('The file exceeds the 2 MB limit.');
      if(!/\.(csv|txt|json)$/i.test(file.name))throw new Error('Select a .csv, .txt, or .json file.');
      const signal=parseSignal(await file.text(),file.name);
      cancelSample();loadInitialSample.current=false;setInputMode('upload');setSourceInfo({is_synthetic:false});setRaw(signal);setSource(file.name);setCode('');setNoise(null);invalidate();
    }catch(e){setError(e.message)}finally{if(input.current)input.current.value=''}
  }
  async function infer(id){
    if(engine==='demo')return {...predictDemo(raw,options,id),pipeline:'demo',input_length:256};
    const descriptor=descriptors.find(d=>d.id===id);
    if(!descriptor||modelsStatus!=='ready')throw new Error('Could not load the model configuration. Check the API connection and retry.');
    if(descriptor.pipeline==='team'&&raw.length!==180)throw new Error(`${MODELS.find(m=>m.id===id)?.name||id} requires exactly 180 raw ECG samples at 360 Hz; this signal contains ${raw.length} samples. Select a real test heartbeat or upload a file with the required input format.`);
    if(['unavailable','error'].includes(descriptor.status))throw new Error(`${MODELS.find(m=>m.id===id)?.name||id} is unavailable: ${descriptor.error||'check the checkpoint on the backend.'}`);
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
    inferenceControllers.current.add(controller);
    try{
      const payload={signal:raw,model:id};
      if(descriptor.pipeline==='demo')payload.preprocessing=options;
      const body=await fetchApiJson(apiUrl,'/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:controller.signal});
      return predictionResult(body,id);
    }catch(e){throw new Error(e.name==='AbortError'?'The request was cancelled or the API did not respond within 15 seconds.':e.message)}finally{clearTimeout(timer);inferenceControllers.current.delete(controller)}
  }
  async function run(compare=false){
    if(busy||sampleBusy||samplePending.current)return;
    const generation=++inferenceGeneration.current;
    setBusy(true);setError('');setResults([]);setModelErrors({});
    try{
      if(compare){
        const settled=await Promise.allSettled(MODELS.map(m=>infer(m.id))),success=[],failures={};
        if(generation!==inferenceGeneration.current)return;
        settled.forEach((entry,i)=>{if(entry.status==='fulfilled')success.push(entry.value);else failures[MODELS[i].id]=entry.reason.message});
        setResults(success);setModelErrors(failures);setPage('compare');
        if(Object.keys(failures).length)setError(`Some models could not run. ${Object.values(failures).join(' ')}`);
      }else{const prediction=await infer(model);if(generation===inferenceGeneration.current)setResults([prediction])}
    }catch(e){if(generation===inferenceGeneration.current)setError(e.message)}finally{if(generation===inferenceGeneration.current)setBusy(false)}
  }
  async function checkApi(){
    connectionController.current?.abort();
    const controller=new AbortController(),config=configRef.current,timer=setTimeout(()=>controller.abort(),8000);
    connectionController.current=controller;setApiStatus('Checking connection…');
    try{
      const health=await fetchApiJson(apiUrl,'/health',{signal:controller.signal});
      if(config!==configRef.current||controller.signal.aborted)return;
      const next=await refreshModels(apiUrl);
      if(!next||config!==configRef.current)return;
      setExamplesReload(value=>value+1);setEvaluationReload(value=>value+1);
      const unavailable=next.filter(d=>['unavailable','error'].includes(d.status)).map(d=>MODELS.find(m=>m.id===d.id)?.name||d.id);
      setApiStatus(`Connected · ${health.mode||health.status||'OK'}${unavailable.length?' · Unavailable: '+unavailable.join(', '):''}`);
    }catch(e){if(config===configRef.current)setApiStatus(`Connection failed: ${e.name==='AbortError'?'The API did not respond within 8 seconds.':e.message}`)}finally{clearTimeout(timer)}
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e=>{e.preventDefault();if(!busy&&!sampleBusy)setPage('workspace')}}><span className="brand-mark"><Activity size={27}/></span><span>ECG<span className="brand-light">studio</span><small>ARRHYTHMIA LAB</small></span></a>
      <nav aria-label="Main navigation">{[{id:'workspace',name:'ECG analysis',icon:LayoutDashboard},{id:'compare',name:'Model comparison',icon:GitCompareArrows},{id:'evaluation',name:'Evaluation results',icon:FlaskConical},{id:'architecture',name:'System architecture',icon:Network}].map(n=><button key={n.id} disabled={busy||sampleBusy} aria-current={page===n.id?'page':undefined} className={page===n.id?'nav-item active':'nav-item'} onClick={()=>setPage(n.id)}><n.icon size={19}/>{n.name}{page===n.id&&<span className="nav-indicator"/>}</button>)}</nav>
    </aside>
    <main>
      <header className="topbar"><div className="breadcrumbs">ECG project <span>/</span><b>{page==='workspace'?'Signal analysis':page==='compare'?'Model comparison':page==='evaluation'?'Evaluation':'Architecture'}</b></div><div className="topbar-actions"><button aria-label="Model connection" aria-haspopup="dialog" aria-expanded={apiOpen} disabled={busy||sampleBusy} className="icon-button settings-button" onClick={()=>setApiOpen(true)}><Settings2 size={17} aria-hidden="true"/><span>Model connection</span></button></div></header>
      <div className="main-content">{page==='architecture'?<Architecture/>:page==='evaluation'?<Evaluation data={evaluation} status={evaluationStatus} error={evaluationError} retry={()=>setEvaluationReload(value=>value+1)}/>:<>
        <div className="page-title"><div><h1>{page==='compare'?'One signal. Three models.':'Heartbeat classification'}</h1><p>{page==='compare'?'Compare predictions from three models using the same input signal.':'Explore ECG signals and classify individual heartbeats.'}</p></div><span className="outline-badge"><Layers size={15}/>5 heartbeat classes</span></div>
        <div className="demo-notice"><Info size={17}/><p>{engine==='demo'?<>You are using a <b>browser-based demonstration model</b>. Its illustrative scores are uncalibrated and are not predictions from a trained checkpoint.</>:apiOnlyDemo?<>This API is running <b>demonstration models</b>. Configure trained checkpoints on the backend to run predictions with the trained models.</>:<>The API runs trained checkpoints on your selected signal. Use <b>real test-set heartbeats</b> to compare predictions with ground-truth labels. Softmax scores are uncalibrated.</>}</p></div>
        {engine==='api'&&modelsStatus==='loading'&&<p className="api-inline-status" role="status">Loading model configurations from the API…</p>}
        {engine==='api'&&modelsStatus==='error'&&<div className="error-banner" role="alert"><AlertCircle size={19}/><span>Could not load API configuration: {modelsError}</span><button onClick={()=>setApiOpen(true)}>Model connection</button></div>}
        {error&&<div className="error-banner" role="alert"><AlertCircle size={19}/><span>{error}</span><button aria-label="Dismiss error" onClick={()=>setError('')}><X size={17}/></button></div>}
        <div className="analysis-grid">
          <section className="card input-card">
            <div className="card-head"><h2>Select a signal</h2><Activity size={20} className="muted-icon"/></div>
            <div className="source-tabs" role="group" aria-label="ECG source"><button aria-pressed={inputMode==='test'} className={inputMode==='test'?'selected':''} disabled={busy||sampleBusy} onClick={()=>{setInputMode('test');if(examples?.samples.length)loadTestSample(examples.samples.find(sample=>sample.code===testCode)?.index??examples.samples[0].index)}}>Real test set</button><button aria-pressed={inputMode==='synthetic'} className={inputMode==='synthetic'?'selected':''} disabled={busy||sampleBusy} onClick={()=>choose(code||'N')}>Synthetic</button></div>
            {inputMode==='test'?<>
              <label className="field-label" htmlFor="test-class">Ground-truth class</label>
              <div className="select-wrap"><select id="test-class" value={testCode} disabled={busy||sampleBusy||examplesStatus!=='ready'} onChange={event=>{const sample=examples.samples.find(sample=>sample.code===event.target.value);setTestCode(event.target.value);if(sample)loadTestSample(sample.index)}}>{CLASSES.map(c=><option key={c.code} value={c.code}>{c.code} · {c.label_en}</option>)}</select><ChevronDown size={15}/></div>
              <div className="test-index-control"><label className="field-label" htmlFor="test-index">Heartbeat index</label><div><input id="test-index" type="number" min="0" max={(examples?.total||18098)-1} step="1" value={testIndex} disabled={busy||sampleBusy} onChange={event=>setTestIndex(event.target.value)}/><button className="secondary-button" disabled={busy||sampleBusy||engine!=='api'} onClick={()=>loadTestSample(testIndex)}>{sampleBusy?<LoaderCircle size={16} className="spin"/>:'Load sample'}</button></div></div>
              <p className="help-text">{examplesStatus==='loading'?'Loading real test-set heartbeats…':examples?`${new Intl.NumberFormat('en-US').format(examples.total)} heartbeats · 180 samples · 360 Hz`:'Enable FastAPI to load real test samples.'}</p>
              {examplesError&&<p className="sample-error" role="alert">{examplesError}</p>}
            </>:inputMode==='upload'?<p className="field-label">Uploaded ECG file</p>:<>
              <label className="field-label" htmlFor="sample">Synthetic ECG example</label>
              <div className="select-wrap"><select id="sample" value={code} disabled={busy||sampleBusy} onChange={e=>choose(e.target.value)}>{!code&&<option value="">Uploaded file</option>}{CLASSES.map(c=><option key={c.code} value={c.code}>{c.code} · {c.label_en}</option>)}</select><ChevronDown size={15}/></div>
              <div className="sample-chips" role="group" aria-label="Select a synthetic ECG class">{CLASSES.map(c=><button key={c.code} aria-label={`${c.code} · ${c.label_en}`} aria-pressed={code===c.code} disabled={busy||sampleBusy} className={code===c.code?'selected':''} title={c.label_en} onClick={()=>choose(c.code)}><span style={{background:c.color}}/>{c.code}</button>)}</div>
            </>}
            <div className="dropzone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!busy&&!sampleBusy)upload(e.dataTransfer.files[0])}}><input ref={input} type="file" accept=".csv,.txt,.json" onChange={e=>upload(e.target.files[0])} hidden/><button disabled={busy||sampleBusy} onClick={()=>input.current.click()}><span className="upload-icon"><Upload size={21}/></span><b>Upload your ECG</b><span>Drop a file here or browse</span><small>CSV, TXT, JSON · up to 2 MB</small></button></div>
            <div className="source-name"><FileText size={14}/><span title={source}>{source}</span><button className="icon-button" aria-label="Download current signal as CSV" onClick={()=>downloadText('ecg-heartbeat.csv','signal\n'+raw.join('\n'),'text/csv')}><Download size={15}/></button></div>
            {sourceInfo.expected_class&&<div className="ground-truth"><span>Ground truth</span><b>{sourceInfo.expected_class.code} · {sourceInfo.expected_class.label_en}</b><small>For comparison only; this label is not sent to the model.</small></div>}
            {inputMode==='synthetic'&&<><div className="input-divider"/><div className="noise-head"><label htmlFor="noise">Simulated noise</label><span>{noise===null?'Uploaded file':`${new Intl.NumberFormat('en-US',{maximumFractionDigits:1}).format(noise*100)}%`}</span></div><input id="noise" className="noise-slider" type="range" min="0" max="0.2" step="0.005" value={noise??0} disabled={!code||busy} onChange={e=>choose(code,Number(e.target.value))}/><p className="help-text">Applies to synthetic samples only. Upload a single heartbeat rather than a full recording.</p></>}
            {needsRaw180&&<p className={`input-contract ${raw.length!==180?'input-contract-error':''}`}>{raw.length===180?'180 raw samples · trained models require 360 Hz.':'Trained models require exactly 180 raw samples at 360 Hz. Your upload is preserved; select a heartbeat with the required number of samples.'}</p>}
          </section>
          <Waveform raw={raw} processed={processed} mode={chartMode} setMode={setChartMode} noise={noise} inputLength={actualMetadata.input_length} processedPending={processedPending}/>
        </div>
        <section className="card pipeline-card">
          <div className="pipeline-heading"><SlidersHorizontal size={19}/><h2>Signal preprocessing</h2></div>
          {isTeam?<div className="fixed-pipeline"><span className="badge">Fixed pipeline</span><p><b>Band-pass 0.5–40 Hz → Z-score</b><span>180 raw samples at 360 Hz · API preprocessing · no resampling</span></p></div>:apiNotReady?<p className="help-text">Awaiting pipeline configuration from the API. Open Model connection to check the backend.</p>:<div className="pipeline-toggles">{[{id:'detrend',name:'Detrending',sub:'Linear detrending'},{id:'smooth',name:'Smoothing',sub:'Moving average · 5 samples'},{id:'normalize',name:'Normalization',sub:'Z-score'}].map(o=><label key={o.id} className="toggle-item"><input type="checkbox" checked={options[o.id]} disabled={busy||sampleBusy} onChange={e=>{setOptions({...options,[o.id]:e.target.checked});invalidate()}}/><span className="toggle-track"/><span><b>{o.name}</b><small>{o.sub}</small></span></label>)}<div className="resample-label"><CheckCircle2 size={17}/><span><b>256 samples</b><small>Demo resampling</small></span></div></div>}
        </section>
        {page==='workspace'?<>
          <section className="inference-controls" aria-label="Select a model and run a prediction"><div><h2>Classification model</h2></div><div className="model-picker" role="group" aria-label="Select a classification model">{MODELS.map(m=><button key={m.id} aria-pressed={model===m.id} className={model===m.id?'selected':''} disabled={busy||sampleBusy} onClick={()=>{setModel(m.id);invalidate()}}><span className="radio-circle">{model===m.id&&<i/>}</span><span><b>{m.name}</b><small>{engine==='api'?modelStatus(descriptors.find(d=>d.id===m.id)):m.kind}</small></span></button>)}</div><button className="primary-button" disabled={busy||sampleBusy||apiNotReady} onClick={()=>run()}>{busy?<LoaderCircle className="spin" size={17}/>:<Play size={17}/>} {busy?'Analyzing…':'Run prediction'}</button></section>
          <div className="result-grid">
            <section aria-live="polite" className={`card result-card ${result?'has-result':''}`}>
              <div className="card-head"><h2>Classification result</h2><span className="badge">{result?(result.is_demo?'DEMO':'CHECKPOINT'):'NOT RUN'}</span></div>
              {result?<><div className="result-display"><span className="result-letter" style={{color:classInfo?.color}}>{result.prediction.code}</span><div><h3>{result.prediction.label_en}</h3><p>{classInfo?.description}</p></div></div>{sourceInfo.expected_class&&<p className={`prediction-match ${sourceInfo.expected_class.code===result.prediction.code?'match':'mismatch'}`}>{sourceInfo.expected_class.code===result.prediction.code?<CheckCircle2 size={16}/>:<AlertCircle size={16}/>}Ground truth: {sourceInfo.expected_class.code} · {sourceInfo.expected_class.code===result.prediction.code?'Prediction matches ground truth':'Prediction differs from ground truth'}</p>}<div className="confidence-display"><span>{result.is_demo?'Illustrative score':'Softmax score'}</span><strong>{(result.confidence*100).toFixed(1)}<small>%</small></strong></div><div className="confidence-track"><span style={{width:`${result.confidence*100}%`,background:classInfo?.color}}/></div><p className="help-text score-note">The score for the predicted class is not calibrated as a clinical probability.</p><p className="result-pipeline">{pipelineLabel(result,selectedDescriptor)}</p><div className="result-foot"><span>{MODELS.find(m=>m.id===result.model_id)?.name} · {result.inference_ms.toFixed(2)} ms</span><JsonExportButton filename="ecg-result.json" getData={()=>resultExport(result,{source,sourceInfo,raw,descriptor:selectedDescriptor,options})}/></div></>:<div className="empty-result"><span><Activity size={30}/></span><h3>Ready to analyze</h3><p>Select a heartbeat and run a prediction to view its class.</p></div>}
            </section>
            <section className="card probability-card"><div className="card-head"><h2>{result?.is_demo?'Illustrative scores':'Softmax scores'}</h2><span className="muted-label">N / S / V / F / Q</span></div><div className="probability-bars">{CLASSES.map(c=>{const p=result?.probabilities.find(p=>p.code===c.code)?.probability;return <div className="probability-row" key={c.code}><span className="class-code" style={{color:c.color}}>{c.code}</span><div><div className="probability-label"><span>{c.label_en}</span><b>{p===undefined?'—':`${(p*100).toFixed(1)}%`}</b></div><div className="probability-track"><span style={{width:p===undefined?'0%':`${p*100}%`,background:c.color}}/></div></div></div>})}</div></section>
          </div>
        </>:<section className="card compare-card">
          <div className="card-head"><h2>Prediction comparison</h2><div className="compare-actions"><JsonExportButton className="secondary-button" disabled={!results.length} filename="ecg-comparison.json" getData={()=>results.map(result=>resultExport(result,{source,sourceInfo,raw,descriptor:descriptors.find(d=>d.id===result.model_id),options}))}/><button className="primary-button" disabled={busy||sampleBusy||engine==='api'&&modelsStatus!=='ready'} onClick={()=>run(true)}>{busy?<LoaderCircle size={17} className="spin"/>:<Play size={17}/>} {busy?'Comparing…':'Run all 3 models'}</button></div></div>
          <div className="comparison-grid">{MODELS.map(m=>{const r=results.find(r=>r.model_id===m.id),descriptor=engine==='demo'?demoDescriptor:descriptors.find(d=>d.id===m.id);return <div className="comparison-model" key={m.id}>
            <div className="comparison-head"><Layers size={20}/><h3>{m.name}</h3></div><p>{engine==='demo'?m.note:modelStatus(descriptor)}</p><span className="comparison-class" style={{color:CLASSES.find(c=>c.code===r?.prediction.code)?.color}}>{r?.prediction.code||'—'}</span><b>{r?.prediction.label_en||(modelErrors[m.id]?'Prediction unavailable':'Awaiting prediction')}</b>
            {r&&sourceInfo.expected_class&&<p className={`prediction-match ${r.prediction.code===sourceInfo.expected_class.code?'match':'mismatch'}`}>Ground truth {sourceInfo.expected_class.code} · {r.prediction.code===sourceInfo.expected_class.code?'Match':'Mismatch'}</p>}
            <div className="comparison-metrics"><span>{r?.is_demo?'Illustrative score':'Softmax score'}<strong>{r?`${(r.confidence*100).toFixed(1)}%`:'—'}</strong></span><span>Inference<strong>{r?`${r.inference_ms.toFixed(2)} ms`:'—'}</strong></span></div>
            <small>{r?(r.is_demo?'Demonstration prototype · untrained':'Trained checkpoint'):'Same input signal'}</small><p className="result-pipeline">{pipelineLabel(r,descriptor)}</p>{modelErrors[m.id]&&<p className="comparison-error">{modelErrors[m.id]}</p>}
          </div>})}</div>
          <p className="help-text compare-help">Models may use different pipelines; check the input length and pipeline for each result. Preprocessing options apply only to demonstration models; trained checkpoints use the fixed API pipeline. Each time shown is a single inference measurement, not a standardized benchmark. Demo scores should not be used to evaluate trained models.</p>
        </section>}
        <div className="footnote"><Info size={14}/><span>For research and presentations. Results are not intended for medical diagnosis.</span><button disabled={busy||sampleBusy} onClick={()=>setPage(page==='compare'?'workspace':'compare')}>{page==='compare'?'Back to analysis':'Compare 3 models'}<GitCompareArrows size={15}/></button></div>
      </>}</div>
      <footer><span>ECG Studio <b>v1.0</b></span><span>ECG heartbeat classification</span></footer>
    </main>
    {apiOpen&&<div className="modal-backdrop" onClick={()=>setApiOpen(false)}><section ref={modalRef} className="modal card" role="dialog" aria-modal="true" aria-labelledby="modal-title" onClick={e=>e.stopPropagation()}>
      <div className="card-head"><h2 id="modal-title">Model connection</h2><button className="icon-button" aria-label="Close settings" onClick={()=>setApiOpen(false)}><X size={20}/></button></div>
      <p>Browser demo mode runs locally in your browser. FastAPI mode loads each model's configuration and runs trained checkpoints on the backend.</p>
      <label className="field-label" htmlFor="engine">Execution mode</label><select id="engine" value={engine} disabled={busy||sampleBusy} onChange={e=>{setEngine(e.target.value);invalidate()}}><option value="demo">Browser demo</option><option value="api">FastAPI</option></select>
      <label className="field-label" htmlFor="api-url">FastAPI URL</label><input id="api-url" type="url" value={apiUrl} disabled={busy||sampleBusy} onChange={e=>{setApiUrl(e.target.value);setApiStatus('');invalidate()}}/>
      <p className="help-text">The API URL and execution mode are saved on this device. An HTTPS website requires an HTTPS API with compatible CORS settings. To use a local API, run the web app at http://127.0.0.1:5173.</p>
      <div className="modal-actions"><button className="secondary-button" onClick={checkApi}>Test connection</button><button className="primary-button" onClick={()=>setApiOpen(false)}>Apply</button></div>
      {apiStatus&&<p role="status" className="api-status">{apiStatus}</p>}
      {descriptors.length>0&&<ul className="api-model-list">{MODELS.map(m=>{const d=descriptors.find(item=>item.id===m.id);return <li key={m.id}><b>{m.name}</b><span>{modelStatus(d)}</span></li>})}</ul>}
    </section></div>}
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
