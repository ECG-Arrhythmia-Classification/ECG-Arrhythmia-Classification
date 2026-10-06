import test from 'node:test';
import assert from 'node:assert/strict';
import {testExample, predictionResult, resultExport} from '../src/api.mjs';

const sample = () => ({signal: Array.from({length: 180}, (_, i) => Math.sin(i / 20)), index: 42, expected_class: {code: 'V', label_vi: 'Ngoại tâm thu thất'}, sample_rate: 360, is_synthetic: false, source: 'MIT-BIH test'});
const prediction = () => ({model_id: 'cnn', model_name: 'CNN', backend: 'native_pytorch', is_demo: false, prediction: {code: 'N', label_vi: 'Bình thường'}, confidence: 0.8, inference_ms: 1.5, probabilities: [{code: 'N', probability: 0.8}, ...['S', 'V', 'F', 'Q'].map(code => ({code, probability: 0.05}))]});

test('real example requires raw team shape, sampling rate and independent ground truth', () => {
  const normalized = testExample(sample());
  assert.equal(normalized.expected_class.code, 'V');
  assert.equal(normalized.expected_class.label_en, 'Ventricular ectopic beat');
  assert.equal('label_vi' in normalized.expected_class, false);
  assert.deepEqual(normalized.signal, sample().signal);
  for (const change of [{signal: [1, 2]}, {sample_rate: 250}, {is_synthetic: true}, {expected_class: {code: 'X'}}, {signal: Array(180).fill(NaN)}]) {
    assert.throws(() => testExample({...sample(), ...change}));
  }
});

test('API predictions must have a consistent complete probability distribution', () => {
  const normalized = predictionResult(prediction(), 'cnn');
  assert.equal(normalized.prediction.code, 'N');
  assert.equal(normalized.prediction.label_en, 'Normal beat group');
  assert.equal('label_vi' in normalized.prediction, false);
  assert.deepEqual(normalized.probabilities.map(row => row.probability), prediction().probabilities.map(row => row.probability));
  for (const change of [{model_id: 'rnn'}, {confidence: 1.2}, {inference_ms: -1}, {probabilities: prediction().probabilities.slice(0, 4)}, {prediction: {code: 'V'}}, {probabilities: prediction().probabilities.map(p => ({...p, probability: 0.5}))}]) {
    assert.throws(() => predictionResult({...prediction(), ...change}, 'cnn'));
  }
});

test('exports preserve mismatches and unknown labels rather than inventing correctness', () => {
  const signal = sample();
  const exported = resultExport({...prediction(), checkpoint: {path: 'results/cnn/best_cnn_model.pt', sha256: 'verified-api-digest'}}, {source: signal.source, sourceInfo: signal, raw: signal.signal, descriptor: {checkpoint: 'descriptor-fallback.pt'}, options: {}});
  assert.equal(exported.expected_class.code, 'V');
  assert.equal(exported.expected_class.label_en, 'Ventricular ectopic beat');
  assert.equal('label_vi' in exported.expected_class, false);
  assert.equal('label_vi' in exported.result.prediction, false);
  assert.equal(exported.result.prediction.code, 'N');
  assert.equal(exported.matches_expected, false);
  assert.equal(exported.input.test_index, 42);
  assert.equal(exported.model.is_demo, false);
  assert.equal(exported.model.checkpoint.sha256, 'verified-api-digest');
  assert.deepEqual(exported.input.signal, signal.signal);
  const uploaded = resultExport(prediction(), {source: 'heartbeat.csv', sourceInfo: {is_synthetic: false}, raw: signal.signal, options: {}});
  assert.equal(uploaded.matches_expected, null);
  assert.equal(uploaded.expected_class, null);
  assert.equal(uploaded.input.sample_rate, null);
});
