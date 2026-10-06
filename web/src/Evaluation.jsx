import React, {useState} from 'react';
import {Info, RefreshCw} from 'lucide-react';
import JsonExportButton from './JsonExportButton.jsx';

const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—';
const number = value => Number.isFinite(value) ? new Intl.NumberFormat('en-US').format(value) : '—';

export default function Evaluation({data, status, error, retry}) {
  const [selected, setSelected] = useState('cnn');
  const models = data?.models || [];
  const current = models.find(model => model.id === selected) || models[0];
  const best = models.length ? models.reduce((a, b) => a.macro_f1 > b.macro_f1 ? a : b) : null;
  return <div className="evaluation-view">
    <div className="page-title"><div><h1>Evaluation results</h1><p>Compare three trained models on the same project test set.</p></div><button className="secondary-button" disabled={status === 'loading'} onClick={retry}><RefreshCw size={16}/>Reload</button></div>
    <div className="demo-notice"><Info size={17}/><p>These evaluation results are saved in the repository. Predictions on the ECG analysis page run through the API; opening this page does not rerun the experiments.</p></div>
    {status === 'loading' && <p className="api-inline-status" role="status">Loading evaluation results…</p>}
    {error && <div className="error-banner" role="alert"><Info size={18}/><span>{error}</span><button onClick={retry}>Retry</button></div>}
    {!data && status !== 'loading' && <section className="card text-card"><h2>Connect to the API to view evaluation results</h2><p>The API serves the saved test, benchmark, and noise robustness results. Select FastAPI in Model connection, then reload this page.</p></section>}
    {data && <>
      <section className="card evaluation-card">
        <div className="card-head"><h2>Evaluation on {number(data.test_samples)} heartbeats</h2><JsonExportButton className="diagram-download" filename="ecg-evaluation.json" getData={() => data}/></div>
        <div className="evaluation-table-wrap"><table className="evaluation-table"><caption className="sr-only">Clean test-set evaluation. Precision, recall, and F1 are macro-averaged.</caption><thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">Macro F1</th><th scope="col">Weighted F1</th></tr></thead><tbody>{models.map(model => <tr key={model.id}><th scope="row">{model.name}</th><td>{percent(model.accuracy)}</td><td>{percent(model.precision)}</td><td>{percent(model.recall)}</td><td><b>{Number.isFinite(model.macro_f1) ? model.macro_f1.toFixed(4) : '—'}</b></td><td>{Number.isFinite(model.weighted_f1) ? model.weighted_f1.toFixed(4) : '—'}</td></tr>)}</tbody></table></div>
        <p className="help-text evaluation-note">{best && `${best.name} has the highest macro F1 in the saved results (${best.macro_f1.toFixed(4)}). `}Macro F1 gives each class equal weight. Accuracy may be high when class N dominates the dataset. These metrics do not guarantee correct predictions for individual heartbeats.</p>
      </section>
      <section className="card evaluation-card">
        <div className="card-head"><h2>Noise robustness</h2><span className="badge">Gaussian · SNR</span></div>
        <div className="evaluation-table-wrap"><table className="evaluation-table"><caption>Macro F1 · Lower SNR indicates stronger noise</caption><thead><tr><th scope="col">Model</th><th scope="col">Clean</th>{[30, 20, 10].map(snr => <th scope="col" key={snr}>{snr} dB</th>)}</tr></thead><tbody>{models.map(model => <tr key={model.id}><th scope="row">{model.name}</th><td>{model.macro_f1.toFixed(4)}</td>{[30, 20, 10].map(snr => <td key={snr}>{model.robustness?.find(row => row.snr_db === snr)?.macro_f1?.toFixed(4) ?? '—'}</td>)}</tr>)}</tbody></table></div>
      </section>
      <div className="evaluation-detail-grid">
        <section className="card evaluation-card">
          <div className="card-head"><h2>Confusion matrix</h2></div>
          <label className="field-label" htmlFor="confusion-model">Select a model</label><select id="confusion-model" value={current?.id || ''} onChange={event => setSelected(event.target.value)}>{models.map(model => <option key={model.id} value={model.id}>{model.name}</option>)}</select>
          {current?.confusion_matrix ? <div className="evaluation-table-wrap confusion-wrap"><table className="evaluation-table confusion-table"><caption>Rows: ground truth · Columns: predicted labels</caption><thead><tr><th scope="col">Actual / Predicted</th>{(current.class_order || ['N', 'S', 'V', 'F', 'Q']).map(code => <th key={code} scope="col">{code}</th>)}</tr></thead><tbody>{current.confusion_matrix.map((row, i) => <tr key={i}><th scope="row">{current.class_order?.[i] || ['N', 'S', 'V', 'F', 'Q'][i]}</th>{row.map((value, j) => <td className={i === j ? 'matrix-correct' : undefined} key={j}>{number(value)}</td>)}</tr>)}</tbody></table></div> : <p className="help-text">No confusion matrix is available for this model.</p>}
        </section>
        <section className="card evaluation-card"><div className="card-head"><h2>Inference performance</h2></div><div className="evaluation-table-wrap"><table className="evaluation-table"><thead><tr><th scope="col">Model</th><th scope="col">Parameters</th><th scope="col">ms / sample</th></tr></thead><tbody>{models.map(model => <tr key={model.id}><th scope="row">{model.name}</th><td>{number(model.parameters)}</td><td>{model.inference_ms_per_sample?.toFixed(4) ?? '—'}</td></tr>)}</tbody></table></div><p className="help-text evaluation-note">The benchmark reports average inference time per sample using batch inference. This differs from the latency of a web prediction request. Training times are not available for all three models in the saved results.</p></section>
      </div>
    </>}
  </div>;
}
