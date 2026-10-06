import React, {useEffect, useId, useRef, useState} from 'react';
import {createPortal} from 'react-dom';
import {Copy, Download, X} from 'lucide-react';
import {downloadText} from './signal.mjs';

export default function JsonExportButton({filename, getData, disabled, className, children = 'Export JSON'}) {
  const [content, setContent] = useState(null), [feedback, setFeedback] = useState('');
  const trigger = useRef(null), dialog = useRef(null), headingId = useId(), helpId = useId();
  useEffect(() => {
    if (content === null) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog.current.querySelector('button')?.focus();
    return () => {document.body.style.overflow = previousOverflow; trigger.current?.focus();};
  }, [content]);

  const close = () => {setContent(null); setFeedback('');};
  const open = () => {setContent(JSON.stringify(getData(), null, 2)); setFeedback('');};
  async function copy() {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(content);
      setFeedback('JSON copied to the clipboard.');
    } catch {
      setFeedback('Clipboard access is unavailable. Select the JSON below and press Ctrl+C, or use the Copy command.');
    }
  }
  function onKeyDown(event) {
    if (event.key === 'Escape') {event.preventDefault(); event.stopPropagation(); close();}
    if (event.key === 'Tab') {
      const controls = [...dialog.current.querySelectorAll('button,[tabindex="0"]')].filter(control => !control.disabled);
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {event.preventDefault(); last.focus();}
      else if (!event.shiftKey && document.activeElement === last) {event.preventDefault(); first.focus();}
    }
  }
  return <>
    <button ref={trigger} disabled={disabled} className={className} onClick={open}><Download size={15}/>{children}</button>
    {content !== null && createPortal(<div className="modal-backdrop" onClick={close}>
      <section ref={dialog} className="modal card json-export-dialog" role="dialog" aria-modal="true" aria-labelledby={headingId} aria-describedby={helpId} onKeyDown={onKeyDown} onClick={event => event.stopPropagation()}>
        <div className="card-head"><h2 id={headingId}>JSON preview</h2><button className="icon-button" aria-label="Close JSON preview" onClick={close}><X size={20}/></button></div>
        <p id={helpId}>Preview the data to be exported. If your browser blocks the download, copy or select the JSON below.</p>
        <div className="json-export-actions"><button className="primary-button" onClick={() => downloadText(filename, content)}><Download size={16}/>Download JSON</button><button className="secondary-button" onClick={copy}><Copy size={16}/>Copy JSON</button></div>
        {feedback && <p className="json-export-feedback" role="status">{feedback}</p>}
        <pre className="json-export-content" tabIndex="0" aria-label={`Contents of ${filename}`}>{content}</pre>
      </section>
    </div>, document.body)}
  </>;
}
