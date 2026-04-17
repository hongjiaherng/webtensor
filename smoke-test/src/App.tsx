import { useEffect, useState } from 'react';
import {
  detectBackends,
  runCpuTests,
  runWasmTests,
  runWebGpuTests,
  type BackendAvailability,
  type TestResult,
} from './tests';

type BackendResults = { label: string; results: TestResult[] };

function ResultRow({ result }: { result: TestResult }) {
  const detail = result.error
    ? result.error
    : result.passed
      ? `[${result.got?.join(', ')}]`
      : `got [${result.got?.join(', ')}] · expected [${result.expected?.join(', ')}]`;

  return (
    <tr>
      <td className={`status ${result.passed ? 'pass' : 'fail'}`}>{result.passed ? '✓' : '✗'}</td>
      <td>{result.name}</td>
      <td className="detail">{detail}</td>
    </tr>
  );
}

function BackendSection({ label, results }: BackendResults) {
  const passed = results.filter((r) => r.passed).length;
  const allPass = passed === results.length;
  return (
    <section>
      <h2>
        {label}
        <span className={`badge ${allPass ? 'pass' : 'fail'}`}>
          {passed}/{results.length} passed
        </span>
      </h2>
      <table>
        <tbody>
          {results.map((r) => (
            <ResultRow key={r.name} result={r} />
          ))}
        </tbody>
      </table>
    </section>
  );
}

function EnvPanel({ backends }: { backends: BackendAvailability[] }) {
  return (
    <div className="env">
      {backends.map((b) => (
        <div key={b.name} className="env-row">
          <span className="label">{b.name}</span>
          <span className={b.available ? 'pass' : 'fail'}>{b.available ? '✓' : '✗'}</span>
          <span className="detail">{b.detail}</span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<BackendResults[]>([]);
  const [env, setEnv] = useState<BackendAvailability[]>([]);

  useEffect(() => {
    detectBackends().then(setEnv);
  }, []);

  async function run() {
    setRunning(true);
    setResults([]);
    const [cpu, wasm, webgpu] = await Promise.all([
      runCpuTests(),
      runWasmTests(),
      runWebGpuTests(),
    ]);
    setResults([
      { label: 'CPU', results: cpu },
      { label: 'WASM', results: wasm },
      { label: 'WebGPU', results: webgpu },
    ]);
    setRunning(false);
  }

  return (
    <main>
      <h1>webtensor smoke test</h1>
      <EnvPanel backends={env} />
      <section>
        <button onClick={run} disabled={running}>
          {running ? 'Running…' : 'Run tests'}
        </button>
      </section>
      {results.map(({ label, results }) => (
        <BackendSection key={label} label={label} results={results} />
      ))}
    </main>
  );
}
