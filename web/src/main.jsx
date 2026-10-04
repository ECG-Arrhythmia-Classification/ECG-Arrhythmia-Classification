import React,{useState,useRef,useEffect,useMemo} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity,LayoutDashboard,GitCompareArrows,Network,FileText,Upload,Download,Play,ChevronDown,Info,FlaskConical,SlidersHorizontal,CheckCircle2,AlertCircle,LoaderCircle,Maximize2,Layers,Settings2,X} from 'lucide-react';
import {CLASSES,MODELS,syntheticBeat,preprocess,predictDemo,parseSignal,downloadText,statistics} from './signal.mjs';
import {DEFAULT_API_URL,readSetting,saveSetting,fetchApiJson,modelDescriptors,pipelineMetadata,pipelineLabel,modelStatus} from './api.mjs';
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
    <div className="card-head"><h2>Tín hiệu ECG</h2><div className="segmented" role="group" aria-label="Chọn tín hiệu gốc hoặc đã xử lý">
      <button aria-pressed={mode==='raw'} className={mode==='raw'?'active':''} onClick={()=>setMode('raw')}>Gốc</button>
      <button aria-pressed={mode==='processed'} className={mode==='processed'?'active':''} onClick={()=>setMode('processed')}>Đã xử lý</button>
    </div></div>
    <div className="chart-meta">
      <span><i className="line-key" style={{background:signalColor}}/>{mode==='processed'&&processedPending?'Chờ xử lý từ API · tín hiệu gốc':showProcessed?'Biên độ sau tiền xử lý':'Biên độ đầu vào'}</span>
      <span className="chart-readout" aria-live="off">{activeIndex===null?'Di chuyển để xem mẫu':`Mẫu ${activeIndex} · ${data[activeIndex].toFixed(3)}`}</span>
      <button className="icon-button" aria-label={zoom?'Thu nhỏ biểu đồ':'Phóng to biểu đồ'} onClick={()=>{setZoom(!zoom);setHover(null)}}><Maximize2 size={15}/>{zoom?'2×':'1×'}</button>
    </div>
    <div className="chart-wrap">
      <span className="axis-label">{showProcessed?'Biên độ sau xử lý':'Biên độ (đơn vị đầu vào)'}</span>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" tabIndex="0" aria-keyshortcuts="ArrowLeft ArrowRight Home End" aria-describedby="waveform-summary" role="img" aria-label={`Đồ thị ECG ${showProcessed?'đã xử lý':'gốc'} với ${data.length} điểm`}
        onKeyDown={e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?data.length-1:Math.max(0,Math.min(data.length-1,(activeIndex??0)+(e.key==='ArrowRight'?1:-1)));setHover(next);setKeyboardReadout(`Mẫu ${next}, biên độ ${data[next].toFixed(3)}`)}}}
        onPointerLeave={()=>setHover(null)} onPointerMove={e=>{const r=e.currentTarget.getBoundingClientRect(),px=(e.clientX-r.left)/r.width*W;setHover(Math.max(0,Math.min(data.length-1,Math.round((px-pad)/(W-pad*2)*(data.length-1)))))}}>
        <defs><pattern id="minor-grid" width="18" height="18" patternUnits="userSpaceOnUse"><path d="M 18 0 L 0 0 0 18" fill="none" stroke="#21414a" strokeWidth="1"/></pattern><pattern id="major-grid" width="90" height="90" patternUnits="userSpaceOnUse"><rect width="90" height="90" fill="url(#minor-grid)"/><path d="M 90 0 L 0 0 0 90" fill="none" stroke="#34565e" strokeWidth="1"/></pattern></defs>
        <rect width={W} height={H} fill="url(#major-grid)"/><path d={path} stroke={signalColor} strokeWidth="2.5" vectorEffect="non-scaling-stroke" fill="none" strokeLinejoin="round"/>
        {activeIndex!==null&&<g><line x1={x(activeIndex)} x2={x(activeIndex)} y1="0" y2={H} stroke="#96b4bf" strokeDasharray="4 4"/><circle cx={x(activeIndex)} cy={y(data[activeIndex])} r="5" fill="#eafaf5"/></g>}
      </svg>
      <p id="waveform-summary" className="sr-only">{`ECG gồm ${data.length} mẫu hiển thị, biên độ nhỏ nhất ${low.toFixed(3)}, lớn nhất ${high.toFixed(3)}. Dùng phím trái/phải để xem từng mẫu; Home và End đến đầu hoặc cuối tín hiệu.`}</p>
      <span className="sr-only" role="status" aria-live="polite" aria-atomic="true">{keyboardReadout}</span>
      <div className="x-labels"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
      <p className="x-axis-title">Vị trí tương đối trong heartbeat {zoom?'· vùng phóng to':''}</p>
      {mode==='processed'&&processedPending&&<p className="chart-pending">Chạy dự đoán để xem tín hiệu đã xử lý bởi pipeline của API.</p>}
    </div>
    <div className="signal-stats">
      <div><span>Số mẫu gốc</span><strong>{raw.length}</strong></div>
      <div><span>Đầu vào model</span><strong>{inputLength||'—'} <small>mẫu</small></strong></div>
      <div><span>Độ lệch chuẩn</span><strong>{statistics(values).std.toFixed(3)}</strong></div>
      <div><span>Nhiễu mô phỏng</span><strong>{noise===null?'—':`${new Intl.NumberFormat('vi-VN',{maximumFractionDigits:1}).format(noise*100)}%`}</strong></div>
    </div>
  </section>;
}

function Architecture(){return <div className="architecture-view">
  <div className="page-title"><div><h1>Từ tín hiệu đến dự đoán</h1><p>Luồng xử lý ECG từ tín hiệu đầu vào đến kết quả phân loại.</p></div><a className="diagram-download" href={`${import.meta.env.BASE_URL}architecture.svg`} download><Download size={16}/>Tải sơ đồ SVG</a></div>
  <section className="card architecture-card">
    <div className="card-head"><h2>Kiến trúc hệ thống</h2><span className="badge">React + FastAPI</span></div>
    <div className="arch-flow">
      <div className="arch-node"><Upload/><b>ECG đầu vào</b><span>CSV / JSON / mẫu tổng hợp</span></div><div className="flow-line"/>
      <div className="arch-node"><SlidersHorizontal/><b>Preprocessing</b><span>Demo: 256 mẫu<br/>Checkpoint: 180 mẫu</span></div><div className="flow-line"/>
      <div className="arch-models">{MODELS.map(m=><div key={m.id}><Layers size={17}/><b>{m.name}</b></div>)}</div><div className="flow-line"/>
      <div className="arch-node"><CheckCircle2/><b>Evaluation</b><span>Test set · F1 · robustness</span></div><div className="flow-line"/>
      <div className="arch-node final-node"><Activity/><b>Web demo</b><span>Nhãn · xác suất · so sánh</span></div>
    </div>
    <div className="arch-note"><Info size={18}/><p>Demo dùng detrend, làm mượt, resample về 256 mẫu và Z-score tùy chọn. Checkpoint dùng pipeline chung cố định: ECG gốc 180 mẫu tại 360 Hz → bandpass 0,5–40 Hz → Z-score; không resample. Cấu hình từng model được đọc từ FastAPI.</p></div>
  </section>
  <div className="two-col">
    <section className="card text-card"><h2>Hợp đồng đầu vào</h2><ul>
      <li>Một heartbeat, các giá trị hữu hạn; CSV một cột hoặc cột có tiêu đề <code>signal</code>.</li>
      <li>JSON dạng <code>[...]</code> hoặc <code>{'{"signal": [...]}'}</code>.</li>
      <li>Demo: 32–10.000 mẫu đầu vào, nội suy về 256 mẫu.</li>
      <li>Checkpoint: đúng 180 mẫu ECG gốc tại 360 Hz, chưa lọc hoặc chuẩn hóa.</li>
      <li>Thống nhất thứ tự lớp: N, S, V, F, Q.</li>
    </ul></section>
    <section className="card text-card"><h2>Hợp đồng tích hợp</h2><ul>
      <li><code>GET /health</code> · trạng thái backend.</li>
      <li><code>GET /models</code> · pipeline, số mẫu và trạng thái từng model.</li>
      <li><code>POST /preprocess</code> · tín hiệu đã xử lý và thống kê.</li>
      <li><code>POST /predict</code> · nhãn, xác suất, thời gian và metadata pipeline.</li>
      <li>Checkpoint lỗi được báo rõ; không tự chuyển sang demo.</li>
    </ul></section>
  </div>
</div>}

function App(){
  const [page,setPage]=useState('workspace'),[code,setCode]=useState('N'),[noise,setNoise]=useState(.025);
  const [raw,setRaw]=useState(()=>syntheticBeat('N')),[source,setSource]=useState('Mẫu tổng hợp · N'),[options,setOptions]=useState(initialOptions);
  const [model,setModel]=useState('cnn'),[chartMode,setChartMode]=useState('raw'),[results,setResults]=useState([]),[modelErrors,setModelErrors]=useState({});
  const [busy,setBusy]=useState(false),[error,setError]=useState(''),[apiOpen,setApiOpen]=useState(false);
  const [engine,setEngine]=useState(()=>readSetting('engine','demo')==='api'?'api':'demo');
  const [apiUrl,setApiUrl]=useState(()=>readSetting('api-url',DEFAULT_API_URL)),[apiStatus,setApiStatus]=useState('');
  const [descriptors,setDescriptors]=useState([]),[modelsStatus,setModelsStatus]=useState('idle'),[modelsError,setModelsError]=useState('');
  const input=useRef(),modalRef=useRef(),modelsRequest=useRef(0),modelsController=useRef(null),connectionController=useRef(null);
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

  const invalidate=()=>{setResults([]);setModelErrors({});setError('')};

  useEffect(()=>saveSetting('engine',engine),[engine]);
  useEffect(()=>saveSetting('api-url',apiUrl),[apiUrl]);
  useEffect(()=>{
    if(!code||!metadataReady||raw.length===syntheticLength)return;
    setRaw(syntheticBeat(code,syntheticLength,noise??.025));
    setSource(`Mẫu tổng hợp · ${code}`);
    setResults([]);setModelErrors({});setError('');
  },[code,noise,raw.length,syntheticLength,metadataReady]);

  async function refreshModels(url){
    modelsController.current?.abort();
    const controller=new AbortController(),request=++modelsRequest.current;
    modelsController.current=controller;
    setModelsStatus('loading');setModelsError('');setResults([]);setModelErrors({});
    const timer=setTimeout(()=>controller.abort(),8000);
    try{
      const body=await fetchApiJson(url,'/models',{signal:controller.signal});
      const next=modelDescriptors(body);
      if(!next.length)throw new Error('API chưa khai báo model nào.');
      if(next.some(d=>!['demo','team'].includes(d.pipeline)||d.input_length!==(d.pipeline==='team'?180:256)))throw new Error('API chưa cung cấp pipeline và số mẫu hợp lệ cho từng model.');
      if(request!==modelsRequest.current||controller.signal.aborted)return null;
      setDescriptors(next);setModelsStatus('ready');
      return next;
    }catch(e){
      if(request!==modelsRequest.current)return null;
      const message=e.name==='AbortError'?'API /models không phản hồi trong 8 giây.':e.message;
      setDescriptors([]);setModelsStatus('error');setModelsError(message);
      throw new Error(message);
    }finally{clearTimeout(timer)}
  }

  useEffect(()=>{
    connectionController.current?.abort();
    setApiStatus('');setDescriptors([]);setModelsError('');
    if(engine!=='api'){
      ++modelsRequest.current;modelsController.current?.abort();setModelsStatus('idle');
      return;
    }
    refreshModels(apiUrl).catch(()=>{});
    const controller=modelsController.current;
    return()=>{controller?.abort();++modelsRequest.current};
  },[engine,apiUrl]);
  useEffect(()=>()=>{modelsController.current?.abort();connectionController.current?.abort()},[]);

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
    setCode(c);setRaw(syntheticBeat(c,syntheticLength,n));setSource(`Mẫu tổng hợp · ${c}`);setNoise(n);invalidate();
  };
  async function upload(file){
    if(!file)return;
    try{
      if(file.size>2*1024*1024)throw new Error('Tệp vượt quá giới hạn 2 MB.');
      if(!/\.(csv|txt|json)$/i.test(file.name))throw new Error('Chọn tệp .csv, .txt hoặc .json.');
      const signal=parseSignal(await file.text(),file.name);
      setRaw(signal);setSource(file.name);setCode('');setNoise(null);invalidate();
    }catch(e){setError(e.message)}finally{if(input.current)input.current.value=''}
  }
  async function infer(id){
    if(engine==='demo')return {...predictDemo(raw,options,id),pipeline:'demo',input_length:256};
    const descriptor=descriptors.find(d=>d.id===id);
    if(!descriptor||modelsStatus!=='ready')throw new Error('Chưa đọc được cấu hình model. Kiểm tra kết nối API rồi thử lại.');
    if(descriptor.pipeline==='team'&&raw.length!==180)throw new Error(`${MODELS.find(m=>m.id===id)?.name||id} cần đúng 180 mẫu ECG gốc tại 360 Hz; tín hiệu hiện có ${raw.length} mẫu. Chọn mẫu tổng hợp 180 điểm hoặc tải heartbeat đúng định dạng.`);
    if(['unavailable','error'].includes(descriptor.status))throw new Error(`${MODELS.find(m=>m.id===id)?.name||id} chưa sẵn sàng: ${descriptor.error||'kiểm tra checkpoint trên backend.'}`);
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
    try{
      const payload={signal:raw,model:id};
      if(descriptor.pipeline==='demo')payload.preprocessing=options;
      const body=await fetchApiJson(apiUrl,'/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:controller.signal});
      if(!body.prediction||!Array.isArray(body.probabilities)||!Number.isFinite(body.confidence)||!Number.isFinite(body.inference_ms))throw new Error('API trả về kết quả sai hợp đồng.');
      return body;
    }catch(e){throw new Error(e.name==='AbortError'?'API không phản hồi trong 15 giây.':e.message)}finally{clearTimeout(timer)}
  }
  async function run(compare=false){
    setBusy(true);setError('');setResults([]);setModelErrors({});
    try{
      await new Promise(resolve=>setTimeout(resolve,180));
      if(compare){
        const settled=await Promise.allSettled(MODELS.map(m=>infer(m.id))),success=[],failures={};
        settled.forEach((entry,i)=>{if(entry.status==='fulfilled')success.push(entry.value);else failures[MODELS[i].id]=entry.reason.message});
        setResults(success);setModelErrors(failures);setPage('compare');
        if(Object.keys(failures).length)setError(`Một số model chưa chạy được. ${Object.values(failures).join(' ')}`);
      }else setResults([await infer(model)]);
    }catch(e){setError(e.message)}finally{setBusy(false)}
  }
  async function checkApi(){
    connectionController.current?.abort();
    const controller=new AbortController(),config=configRef.current,timer=setTimeout(()=>controller.abort(),8000);
    connectionController.current=controller;setApiStatus('Đang kiểm tra…');
    try{
      const health=await fetchApiJson(apiUrl,'/health',{signal:controller.signal});
      if(config!==configRef.current||controller.signal.aborted)return;
      const next=await refreshModels(apiUrl);
      if(!next||config!==configRef.current)return;
      const unavailable=next.filter(d=>['unavailable','error'].includes(d.status)).map(d=>MODELS.find(m=>m.id===d.id)?.name||d.id);
      setApiStatus(`Đã kết nối · ${health.mode||health.status||'OK'}${unavailable.length?' · Chưa sẵn sàng: '+unavailable.join(', '):''}`);
    }catch(e){if(config===configRef.current)setApiStatus(`Chưa kết nối: ${e.name==='AbortError'?'API không phản hồi trong 8 giây.':e.message}`)}finally{clearTimeout(timer)}
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e=>{e.preventDefault();if(!busy)setPage('workspace')}}><span className="brand-mark"><Activity size={27}/></span><span>ECG<span className="brand-light">studio</span><small>ARRHYTHMIA LAB</small></span></a>
      <nav aria-label="Điều hướng chính">{[{id:'workspace',name:'Phân tích ECG',icon:LayoutDashboard},{id:'compare',name:'So sánh model',icon:GitCompareArrows},{id:'architecture',name:'Kiến trúc hệ thống',icon:Network}].map(n=><button key={n.id} disabled={busy} aria-current={page===n.id?'page':undefined} className={page===n.id?'nav-item active':'nav-item'} onClick={()=>setPage(n.id)}><n.icon size={19}/>{n.name}{page===n.id&&<span className="nav-indicator"/>}</button>)}</nav>
    </aside>
    <main>
      <header className="topbar"><div className="breadcrumbs">Dự án ECG <span>/</span><b>{page==='workspace'?'Phân tích tín hiệu':page==='compare'?'So sánh model':'Kiến trúc'}</b></div><div className="topbar-actions"><span className="demo-badge"><FlaskConical size={14}/>{engine==='demo'?'DEMO MODE':'API MODE'}</span><button aria-label="Kết nối model" disabled={busy} className="icon-button settings-button" onClick={()=>setApiOpen(true)}><Settings2 size={17}/><span>Kết nối model</span></button></div></header>
      <div className="main-content">{page==='architecture'?<Architecture/>:<>
        <div className="page-title"><div><h1>{page==='compare'?'Cùng tín hiệu. Ba góc nhìn.':'Phân tích nhịp tim'}</h1><p>{page==='compare'?'Đối chiếu dự đoán của ba model trên cùng tín hiệu đầu vào.':'Khám phá tín hiệu ECG và phân loại từng heartbeat.'}</p></div><span className="outline-badge"><Layers size={15}/>5 nhóm nhịp tim</span></div>
        <div className="demo-notice"><Info size={17}/><p>{engine==='demo'?<>Bạn đang dùng <b>tín hiệu tổng hợp & model minh họa</b>. Điểm xác suất chưa hiệu chuẩn; chỉ dùng để trình diễn dự án.</>:<>Chế độ FastAPI đang bật. Mỗi model sử dụng pipeline được khai báo bởi API; mẫu tổng hợp vẫn chỉ dùng để minh họa.</>}</p></div>
        {engine==='api'&&modelsStatus==='loading'&&<p className="api-inline-status" role="status">Đang đọc cấu hình model từ API…</p>}
        {engine==='api'&&modelsStatus==='error'&&<div className="error-banner" role="alert"><AlertCircle size={19}/><span>Chưa đọc được cấu hình API: {modelsError}</span><button onClick={()=>setApiOpen(true)}>Kết nối model</button></div>}
        {error&&<div className="error-banner" role="alert"><AlertCircle size={19}/><span>{error}</span><button aria-label="Đóng thông báo lỗi" onClick={()=>setError('')}><X size={17}/></button></div>}
        <div className="analysis-grid">
          <section className="card input-card">
            <div className="card-head"><h2>Chọn tín hiệu</h2><Activity size={20} className="muted-icon"/></div>
            <label className="field-label" htmlFor="sample">Mẫu ECG minh họa</label>
            <div className="select-wrap"><select id="sample" value={code} disabled={busy} onChange={e=>choose(e.target.value)}>{!code&&<option value="">Tệp đã tải lên</option>}{CLASSES.map(c=><option key={c.code} value={c.code}>{c.code} · {c.label_vi}</option>)}</select><ChevronDown size={15}/></div>
            <div className="sample-chips" role="group" aria-label="Chọn nhóm ECG mẫu">{CLASSES.map(c=><button key={c.code} aria-label={`${c.code} · ${c.label_vi}`} aria-pressed={code===c.code} disabled={busy} className={code===c.code?'selected':''} title={c.label_vi} onClick={()=>choose(c.code)}><span style={{background:c.color}}/>{c.code}</button>)}</div>
            <div className="dropzone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!busy)upload(e.dataTransfer.files[0])}}><input ref={input} type="file" accept=".csv,.txt,.json" onChange={e=>upload(e.target.files[0])} hidden/><button disabled={busy} onClick={()=>input.current.click()}><span className="upload-icon"><Upload size={21}/></span><b>Tải tín hiệu của bạn</b><span>Kéo thả hoặc chọn tệp</span><small>CSV, TXT, JSON · tối đa 2 MB</small></button></div>
            <div className="source-name"><FileText size={14}/><span title={source}>{source}</span><button className="icon-button" aria-label="Tải tín hiệu hiện tại dạng CSV" onClick={()=>downloadText('ecg-heartbeat.csv','signal\n'+raw.join('\n'),'text/csv')}><Download size={15}/></button></div>
            <div className="input-divider"/><div className="noise-head"><label htmlFor="noise">Nhiễu mô phỏng</label><span>{noise===null?'Tệp nhập':`${new Intl.NumberFormat('vi-VN',{maximumFractionDigits:1}).format(noise*100)}%`}</span></div>
            <input id="noise" className="noise-slider" type="range" min="0" max="0.2" step="0.005" value={noise??0} disabled={!code||busy} onChange={e=>choose(code,Number(e.target.value))}/>
            <p className="help-text">Chỉ áp dụng cho mẫu tổng hợp. Nhập một heartbeat, không phải bản ghi dài.</p>
            {needsRaw180&&<p className={`input-contract ${raw.length!==180?'input-contract-error':''}`}>{raw.length===180?'180 mẫu gốc · yêu cầu 360 Hz cho checkpoint.':'Checkpoint cần đúng 180 mẫu gốc tại 360 Hz. Tệp nhập được giữ nguyên; hãy chọn heartbeat đúng số mẫu.'}</p>}
          </section>
          <Waveform raw={raw} processed={processed} mode={chartMode} setMode={setChartMode} noise={noise} inputLength={actualMetadata.input_length} processedPending={processedPending}/>
        </div>
        <section className="card pipeline-card">
          <div className="pipeline-heading"><SlidersHorizontal size={19}/><h2>Chuẩn bị tín hiệu</h2></div>
          {isTeam?<div className="fixed-pipeline"><span className="badge">Pipeline cố định</span><p><b>Bandpass 0,5–40 Hz → Z-score</b><span>180 mẫu gốc tại 360 Hz · xử lý trên API · không resample</span></p></div>:apiNotReady?<p className="help-text">Chờ cấu hình pipeline từ API. Mở Kết nối model để kiểm tra backend.</p>:<div className="pipeline-toggles">{[{id:'detrend',name:'Loại xu hướng',sub:'Linear detrend'},{id:'smooth',name:'Làm mượt',sub:'Moving average · 5'},{id:'normalize',name:'Chuẩn hóa',sub:'Z-score'}].map(o=><label key={o.id} className="toggle-item"><input type="checkbox" checked={options[o.id]} disabled={busy} onChange={e=>{setOptions({...options,[o.id]:e.target.checked});invalidate()}}/><span className="toggle-track"/><span><b>{o.name}</b><small>{o.sub}</small></span></label>)}<div className="resample-label"><CheckCircle2 size={17}/><span><b>256 mẫu</b><small>Resample demo</small></span></div></div>}
        </section>
        {page==='workspace'?<>
          <section className="inference-controls" aria-label="Chọn model và chạy dự đoán"><div><h2>Model phân loại</h2></div><div className="model-picker" role="group" aria-label="Chọn model phân loại">{MODELS.map(m=><button key={m.id} aria-pressed={model===m.id} className={model===m.id?'selected':''} disabled={busy} onClick={()=>{setModel(m.id);invalidate()}}><span className="radio-circle">{model===m.id&&<i/>}</span><span><b>{m.name}</b><small>{engine==='api'?modelStatus(descriptors.find(d=>d.id===m.id)):m.kind}</small></span></button>)}</div><button className="primary-button" disabled={busy||apiNotReady} onClick={()=>run()}>{busy?<LoaderCircle className="spin" size={17}/>:<Play size={17}/>} {busy?'Đang phân tích…':'Chạy dự đoán'}</button></section>
          <div className="result-grid">
            <section aria-live="polite" className={`card result-card ${result?'has-result':''}`}>
              <div className="card-head"><h2>Kết quả phân loại</h2><span className="badge">{result?(result.is_demo?'DEMO':'CHECKPOINT'):'CHƯA CHẠY'}</span></div>
              {result?<><div className="result-display"><span className="result-letter" style={{color:classInfo?.color}}>{result.prediction.code}</span><div><h3>{result.prediction.label_vi}</h3><p>{classInfo?.label_en}</p></div></div><div className="confidence-display"><span>Điểm xác suất {result.is_demo?'minh họa':''}</span><strong>{(result.confidence*100).toFixed(1)}<small>%</small></strong></div><div className="confidence-track"><span style={{width:`${result.confidence*100}%`,background:classInfo?.color}}/></div><p className="result-pipeline">{pipelineLabel(result,selectedDescriptor)}</p><div className="result-foot"><span>{MODELS.find(m=>m.id===result.model_id)?.name} · {result.inference_ms.toFixed(2)} ms</span><button onClick={()=>downloadText('ecg-result.json',JSON.stringify({source,options:result.preprocessing?.options||options,...result},null,2))}><Download size={15}/>Xuất JSON</button></div></>:<div className="empty-result"><span><Activity size={30}/></span><h3>Sẵn sàng phân tích</h3><p>Chọn model và chạy dự đoán để xem nhóm nhịp tim.</p></div>}
            </section>
            <section className="card probability-card"><div className="card-head"><h2>Phân bố xác suất</h2><span className="muted-label">N / S / V / F / Q</span></div><div className="probability-bars">{CLASSES.map(c=>{const p=result?.probabilities.find(p=>p.code===c.code)?.probability;return <div className="probability-row" key={c.code}><span className="class-code" style={{color:c.color}}>{c.code}</span><div><div className="probability-label"><span>{c.label_vi}</span><b>{p===undefined?'—':`${(p*100).toFixed(1)}%`}</b></div><div className="probability-track"><span style={{width:p===undefined?'0%':`${p*100}%`,background:c.color}}/></div></div></div>})}</div></section>
          </div>
        </>:<section className="card compare-card">
          <div className="card-head"><h2>So sánh dự đoán</h2><button className="primary-button" disabled={busy||engine==='api'&&modelsStatus!=='ready'} onClick={()=>run(true)}>{busy?<LoaderCircle size={17} className="spin"/>:<Play size={17}/>} {busy?'Đang so sánh…':'Chạy cả 3 model'}</button></div>
          <div className="comparison-grid">{MODELS.map(m=>{const r=results.find(r=>r.model_id===m.id),descriptor=engine==='demo'?demoDescriptor:descriptors.find(d=>d.id===m.id);return <div className="comparison-model" key={m.id}>
            <div className="comparison-head"><Layers size={20}/><h3>{m.name}</h3></div><p>{engine==='demo'?m.note:modelStatus(descriptor)}</p><span className="comparison-class" style={{color:CLASSES.find(c=>c.code===r?.prediction.code)?.color}}>{r?.prediction.code||'—'}</span><b>{r?.prediction.label_vi||(modelErrors[m.id]?'Chưa chạy được':'Chờ chạy dự đoán')}</b>
            <div className="comparison-metrics"><span>Điểm xác suất<strong>{r?`${(r.confidence*100).toFixed(1)}%`:'—'}</strong></span><span>Inference<strong>{r?`${r.inference_ms.toFixed(2)} ms`:'—'}</strong></span></div>
            <small>{r?(r.is_demo?'Prototype demo · chưa huấn luyện':'Checkpoint thật'):'Cùng tín hiệu đầu vào'}</small><p className="result-pipeline">{pipelineLabel(r,descriptor)}</p>{modelErrors[m.id]&&<p className="comparison-error">{modelErrors[m.id]}</p>}
          </div>})}</div>
          <p className="help-text compare-help">Các model có thể dùng pipeline khác nhau; xem số mẫu và pipeline của từng kết quả. Tùy chọn tiền xử lý chỉ áp dụng cho model demo; checkpoint dùng pipeline cố định của API. Thời gian là một lần inference thực tế, không phải benchmark chuẩn. Không dùng điểm demo để đánh giá model đã huấn luyện.</p>
        </section>}
        <div className="footnote"><Info size={14}/><span>Demo phục vụ học tập và thuyết trình. Kết quả không dùng để chẩn đoán y tế.</span><button disabled={busy} onClick={()=>setPage(page==='compare'?'workspace':'compare')}>{page==='compare'?'Về phân tích':'So sánh 3 model'}<GitCompareArrows size={15}/></button></div>
      </>}</div>
      <footer><span>ECG Studio <b>v1.0</b></span><span>Phân loại tín hiệu ECG</span></footer>
    </main>
    {apiOpen&&<div className="modal-backdrop" onClick={()=>setApiOpen(false)}><section ref={modalRef} className="modal card" role="dialog" aria-modal="true" aria-labelledby="modal-title" onClick={e=>e.stopPropagation()}>
      <div className="card-head"><h2 id="modal-title">Kết nối model</h2><button className="icon-button" aria-label="Đóng cài đặt" onClick={()=>setApiOpen(false)}><X size={20}/></button></div>
      <p>Demo chạy trong trình duyệt. Chế độ API đọc cấu hình từng model và dùng FastAPI để chạy checkpoint.</p>
      <label className="field-label" htmlFor="engine">Chế độ thực thi</label><select id="engine" value={engine} disabled={busy} onChange={e=>{setEngine(e.target.value);invalidate()}}><option value="demo">Demo trong trình duyệt</option><option value="api">FastAPI / checkpoint</option></select>
      <label className="field-label" htmlFor="api-url">Địa chỉ FastAPI</label><input id="api-url" type="url" value={apiUrl} disabled={busy} onChange={e=>{setApiUrl(e.target.value);setApiStatus('');invalidate()}}/>
      <p className="help-text">Địa chỉ API và chế độ được lưu trên thiết bị này. Web HTTPS cần API HTTPS với CORS phù hợp. Để kết nối API local, chạy web local tại http://127.0.0.1:5173.</p>
      <div className="modal-actions"><button className="secondary-button" onClick={checkApi}>Kiểm tra kết nối</button><button className="primary-button" onClick={()=>setApiOpen(false)}>Áp dụng</button></div>
      {apiStatus&&<p role="status" className="api-status">{apiStatus}</p>}
      {descriptors.length>0&&<ul className="api-model-list">{MODELS.map(m=>{const d=descriptors.find(item=>item.id===m.id);return <li key={m.id}><b>{m.name}</b><span>{modelStatus(d)}</span></li>})}</ul>}
    </section></div>}
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
