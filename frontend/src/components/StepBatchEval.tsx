/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import {
  Upload, FileArchive, FileText, CheckCircle2,
  AlertTriangle, Loader2, BarChart2, Trash2,
  TrendingDown, TrendingUp, Target, Clock,
  ChevronUp, ChevronDown,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface WorstError {
  string1: string;
  string2: string;
  probability: number;
}

interface BatchResult {
  total: number;
  threshold: number;
  above_threshold_count: number;
  below_threshold_count: number;
  mean_probability: number;
  median_probability: number;
  worst_errors: WorstError[];
  time_ms: number;
}

type SortKey = 'probability' | 'string1' | 'string2';
type SortDir = 'asc' | 'desc';

export default function StepBatchEval() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [modelDrag, setModelDrag] = useState(false);
  const [csvDrag, setCsvDrag] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('probability');
  const [sortDir, setSortDir] = useState<SortDir>('asc');

  const modelRef = useRef<HTMLInputElement>(null);
  const csvRef = useRef<HTMLInputElement>(null);

  const handleModelDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setModelDrag(e.type === 'dragenter' || e.type === 'dragover');
  };
  const handleCsvDrag = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setCsvDrag(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleModelDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setModelDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.name.endsWith('.zip')) setModelFile(f);
    else setError('Загрузите ZIP-архив модели.');
  };
  const handleCsvDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation();
    setCsvDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.name.endsWith('.csv')) setCsvFile(f);
    else setError('Загрузите CSV-датасет с двумя колонками.');
  };

  const runBatch = async () => {
    if (!modelFile || !csvFile) {
      setError('Загрузите модель (ZIP) и датасет (CSV).');
      return;
    }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const form = new FormData();
      form.append('file', modelFile);
      form.append('dataset', csvFile);
      const res = await fetch('/api/v1/inference/batch', { method: 'POST', body: form });
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || j.detail || 'Ошибка сервера');
      }
      setResult(await res.json());
    } catch (e: any) {
      setError(e.message || 'Неизвестная ошибка');
    } finally {
      setLoading(false);
    }
  };

  const sortedErrors = result
    ? [...result.worst_errors].sort((a, b) => {
        const va = sortKey === 'probability' ? a.probability : a[sortKey];
        const vb = sortKey === 'probability' ? b.probability : b[sortKey];
        const cmp = typeof va === 'number' && typeof vb === 'number'
          ? va - vb
          : String(va).localeCompare(String(vb), 'ru');
        return sortDir === 'asc' ? cmp : -cmp;
      })
    : [];

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir(key === 'probability' ? 'asc' : 'asc'); }
  };

  const SortIcon = ({ k }: { k: SortKey }) =>
    sortKey === k
      ? (sortDir === 'asc' ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />)
      : <ChevronUp className="h-3.5 w-3.5 opacity-30" />;

  const recallPct = result
    ? (result.above_threshold_count / result.total * 100).toFixed(1)
    : '—';
  const errorPct = result
    ? (result.below_threshold_count / result.total * 100).toFixed(1)
    : '—';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">

      {/* LEFT INFO COLUMN */}
      <div className="col-span-1 lg:col-span-11 xl:col-span-4 flex flex-col justify-center py-6">
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          <div className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold text-violet-700 bg-violet-50 rounded-full border border-violet-100 uppercase tracking-wider">
            Шаг 3 • Пакетная оценка
          </div>

          <h2 className="font-sans text-3xl font-extrabold tracking-tight text-slate-900 leading-tight sm:text-4xl">
            Проверяйте модель <br />
            на целом датасете
          </h2>

          <p className="font-sans text-base text-slate-500 leading-relaxed">
            Загрузите обученную модель и CSV-датасет верных пар. Система прогонит
            каждую пару через модель и покажет статистику — сколько пар выше порога,
            сколько «сомнительных», а также 20 самых низкоуверенных примеров.
          </p>

          <div className="space-y-5 pt-4 border-t border-slate-100">
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-violet-50 text-violet-600 border border-violet-100">
                <Target className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">Реальные метрики</h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Порог берётся из обученной модели — результат честный
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-rose-50 text-rose-600 border border-rose-100">
                <TrendingDown className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">Топ-20 ошибок</h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Пары с наименьшей уверенностью — то, что стоит исправить
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-amber-50 text-amber-600 border border-amber-100">
                <Clock className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">Батч-оптимизация</h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Все строки кодируются за один проход — быстрее в N раз
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* RIGHT MAIN COLUMN */}
      <div className="col-span-1 lg:col-span-12 xl:col-span-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="space-y-6"
        >
          {/* UPLOAD CARD */}
          <div className="bg-white/95 backdrop-blur-md rounded-3xl p-6 sm:p-8 shadow-2xl shadow-violet-500/5 border border-slate-100 space-y-6">

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              {/* Model ZIP */}
              <div className="space-y-2">
                <h3 className="font-sans text-sm font-bold text-slate-900 uppercase tracking-tight flex items-center gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-white text-[10px] font-mono">1</span>
                  Модель (ZIP)
                </h3>
                <div
                  onDragEnter={handleModelDrag} onDragOver={handleModelDrag}
                  onDragLeave={() => setModelDrag(false)} onDrop={handleModelDrop}
                  onClick={() => !modelFile && modelRef.current?.click()}
                  className={`flex flex-col items-center justify-center p-5 border-2 border-dashed rounded-2xl transition-all duration-300 min-h-[100px] ${
                    modelFile ? 'border-emerald-300 bg-emerald-50/20 cursor-default'
                      : modelDrag ? 'border-violet-400 bg-violet-50/40 cursor-pointer'
                      : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/30 cursor-pointer'
                  }`}
                >
                  <input ref={modelRef} type="file" accept=".zip" className="hidden"
                    onChange={e => { if (e.target.files?.[0]) setModelFile(e.target.files[0]); e.target.value = ''; }}
                    disabled={!!modelFile} />
                  {modelFile ? (
                    <div className="flex items-center gap-3 w-full">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-100 text-blue-700 flex-shrink-0">
                        <FileArchive className="h-5 w-5" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-semibold text-slate-800 truncate">{modelFile.name}</p>
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800">
                          <CheckCircle2 className="h-3 w-3" /> Загружена
                        </span>
                      </div>
                      <button onClick={e => { e.stopPropagation(); setModelFile(null); setResult(null); }}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-50 transition-colors">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <Upload className="h-7 w-7 text-slate-300 mb-2" />
                      <p className="text-xs text-slate-400 text-center">Перетащите ZIP-модель<br />или кликните</p>
                    </>
                  )}
                </div>
              </div>

              {/* CSV dataset */}
              <div className="space-y-2">
                <h3 className="font-sans text-sm font-bold text-slate-900 uppercase tracking-tight flex items-center gap-2">
                  <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-white text-[10px] font-mono">2</span>
                  Датасет (CSV)
                </h3>
                <div
                  onDragEnter={handleCsvDrag} onDragOver={handleCsvDrag}
                  onDragLeave={() => setCsvDrag(false)} onDrop={handleCsvDrop}
                  onClick={() => !csvFile && csvRef.current?.click()}
                  className={`flex flex-col items-center justify-center p-5 border-2 border-dashed rounded-2xl transition-all duration-300 min-h-[100px] ${
                    csvFile ? 'border-emerald-300 bg-emerald-50/20 cursor-default'
                      : csvDrag ? 'border-violet-400 bg-violet-50/40 cursor-pointer'
                      : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50/30 cursor-pointer'
                  }`}
                >
                  <input ref={csvRef} type="file" accept=".csv" className="hidden"
                    onChange={e => { if (e.target.files?.[0]) setCsvFile(e.target.files[0]); e.target.value = ''; }}
                    disabled={!!csvFile} />
                  {csvFile ? (
                    <div className="flex items-center gap-3 w-full">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-100 text-violet-700 flex-shrink-0">
                        <FileText className="h-5 w-5" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-semibold text-slate-800 truncate">{csvFile.name}</p>
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800">
                          <CheckCircle2 className="h-3 w-3" /> Загружен
                        </span>
                      </div>
                      <button onClick={e => { e.stopPropagation(); setCsvFile(null); setResult(null); }}
                        className="p-1.5 rounded-lg text-slate-400 hover:text-rose-500 hover:bg-rose-50 transition-colors">
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <Upload className="h-7 w-7 text-slate-300 mb-2" />
                      <p className="text-xs text-slate-400 text-center">Перетащите CSV-файл<br />или кликните</p>
                    </>
                  )}
                </div>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-3 p-4 bg-rose-50 border border-rose-100 rounded-2xl text-rose-700 text-sm">
                <AlertTriangle className="h-5 w-5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <button
              onClick={runBatch}
              disabled={loading || !modelFile || !csvFile}
              className={`w-full flex items-center justify-center gap-3 px-6 py-3.5 rounded-2xl font-semibold text-sm transition-all duration-300 ${
                loading || !modelFile || !csvFile
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  : 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-lg shadow-violet-500/20 hover:shadow-violet-500/30 hover:scale-[1.01]'
              }`}
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Проверяем датасет… Эмбеддинги батч-кодируются
                </>
              ) : (
                <>
                  <BarChart2 className="h-5 w-5" />
                  Проверить датасет
                </>
              )}
            </button>
          </div>

          {/* RESULTS */}
          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.4 }}
                className="space-y-5"
              >
                {/* STAT CARDS */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
                  <div className="bg-white/95 rounded-2xl p-4 border border-slate-100 shadow-sm text-center">
                    <p className="text-xs text-slate-400 font-medium uppercase tracking-wide mb-1">Всего пар</p>
                    <p className="text-2xl font-extrabold text-slate-900">{result.total.toLocaleString('ru')}</p>
                    <p className="text-[10px] text-slate-400 mt-0.5">{(result.time_ms / 1000).toFixed(1)} с</p>
                  </div>

                  <div className="bg-indigo-50/80 rounded-2xl p-4 border border-indigo-100 shadow-sm text-center">
                    <p className="text-xs text-indigo-700 font-medium uppercase tracking-wide mb-1 flex items-center justify-center gap-1">
                      <Target className="h-3.5 w-3.5" /> Порог
                    </p>
                    <p className="text-2xl font-extrabold text-indigo-800">{(result.threshold * 100).toFixed(0)}%</p>
                    <p className="text-[10px] text-indigo-600 mt-0.5">из meta.json модели</p>
                  </div>

                  <div className="bg-emerald-50/80 rounded-2xl p-4 border border-emerald-100 shadow-sm text-center">
                    <p className="text-xs text-emerald-700 font-medium uppercase tracking-wide mb-1 flex items-center justify-center gap-1">
                      <TrendingUp className="h-3.5 w-3.5" /> Распознано
                    </p>
                    <p className="text-2xl font-extrabold text-emerald-800">{recallPct}%</p>
                    <p className="text-[10px] text-emerald-600 mt-0.5">recall — пар ≥ {(result.threshold * 100).toFixed(0)}%</p>
                  </div>

                  <div className={`rounded-2xl p-4 border shadow-sm text-center ${
                    result.below_threshold_count / result.total > 0.05
                      ? 'bg-rose-50/80 border-rose-100'
                      : 'bg-amber-50/80 border-amber-100'
                  }`}>
                    <p className={`text-xs font-medium uppercase tracking-wide mb-1 flex items-center justify-center gap-1 ${
                      result.below_threshold_count / result.total > 0.05 ? 'text-rose-700' : 'text-amber-700'
                    }`}>
                      <TrendingDown className="h-3.5 w-3.5" /> Пропущено
                    </p>
                    <p className={`text-2xl font-extrabold ${
                      result.below_threshold_count / result.total > 0.05 ? 'text-rose-800' : 'text-amber-800'
                    }`}>{errorPct}%</p>
                    <p className={`text-[10px] mt-0.5 ${
                      result.below_threshold_count / result.total > 0.05 ? 'text-rose-600' : 'text-amber-600'
                    }`}>пар &lt; {(result.threshold * 100).toFixed(0)}%</p>
                  </div>

                  <div className="bg-blue-50/80 rounded-2xl p-4 border border-blue-100 shadow-sm text-center">
                    <p className="text-xs text-blue-700 font-medium uppercase tracking-wide mb-1">Ср. вероятность</p>
                    <p className="text-2xl font-extrabold text-blue-800">{(result.mean_probability * 100).toFixed(1)}%</p>
                    <p className="text-[10px] text-blue-600 mt-0.5">медиана {(result.median_probability * 100).toFixed(1)}%</p>
                  </div>
                </div>

                {/* CONFIDENCE DISTRIBUTION */}
                <div className="bg-white/95 rounded-3xl p-5 border border-slate-100 shadow-sm">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="font-sans text-sm font-bold text-slate-900">Уверенность модели</h3>
                      <p className="text-xs text-slate-400 mt-0.5">
                        Сколько пар получили вероятность выше каждой отметки
                      </p>
                    </div>
                    <div className="text-right text-xs text-slate-400 leading-relaxed">
                      Порог recall: <span className="font-bold text-indigo-600">{(result.threshold * 100).toFixed(0)}%</span><br />
                      <span className="text-[10px]">минимум для захвата 95% пар</span>
                    </div>
                  </div>
                  <div className="space-y-3">
                    {[
                      { label: '≥ 90%', value: result.recall_at_90, color: 'bg-emerald-500', note: '← высокая уверенность' },
                      { label: '≥ 70%', value: result.recall_at_70, color: 'bg-blue-400', note: '← хорошая уверенность' },
                      { label: '≥ 50%', value: result.recall_at_50, color: 'bg-indigo-300', note: '← выше случайного' },
                      { label: `≥ ${(result.threshold * 100).toFixed(0)}%`, value: result.above_threshold_count / result.total, color: 'bg-slate-300', note: '← порог recall 95%' },
                    ].map(({ label, value, color, note }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs font-mono font-bold text-slate-600 w-12 text-right flex-shrink-0">{label}</span>
                        <div className="flex-1 h-5 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-700 ${color}`}
                            style={{ width: `${(value * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold text-slate-700 w-12 flex-shrink-0">{(value * 100).toFixed(1)}%</span>
                        <span className="text-[10px] text-slate-400 w-32 flex-shrink-0">{note}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* WORST ERRORS TABLE */}
                <div className="bg-white/95 backdrop-blur-md rounded-3xl border border-slate-100 shadow-xl overflow-hidden">
                  <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
                    <h3 className="font-sans text-sm font-bold text-slate-900 flex items-center gap-2">
                      <TrendingDown className="h-4 w-4 text-rose-500" />
                      Топ-{result.worst_errors.length} пар с наименьшей уверенностью
                    </h3>
                    <span className="text-xs text-slate-400 font-mono">нажмите на заголовок для сортировки</span>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="bg-slate-50/70 text-left">
                          <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide w-16">
                            <button onClick={() => toggleSort('probability')}
                              className="flex items-center gap-1 hover:text-slate-900 transition-colors">
                              % <SortIcon k="probability" />
                            </button>
                          </th>
                          <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">
                            <button onClick={() => toggleSort('string1')}
                              className="flex items-center gap-1 hover:text-slate-900 transition-colors">
                              Строка 1 <SortIcon k="string1" />
                            </button>
                          </th>
                          <th className="px-4 py-3 text-xs font-semibold text-slate-500 uppercase tracking-wide">
                            <button onClick={() => toggleSort('string2')}
                              className="flex items-center gap-1 hover:text-slate-900 transition-colors">
                              Строка 2 <SortIcon k="string2" />
                            </button>
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {sortedErrors.map((e, i) => {
                          const pct = e.probability * 100;
                          const color = pct < 20
                            ? 'text-rose-700 bg-rose-50'
                            : pct < 40
                            ? 'text-amber-700 bg-amber-50'
                            : 'text-slate-700 bg-slate-50';
                          return (
                            <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                              <td className="px-4 py-3">
                                <span className={`inline-flex items-center px-2 py-0.5 rounded-lg text-xs font-bold font-mono ${color}`}>
                                  {pct.toFixed(1)}%
                                </span>
                              </td>
                              <td className="px-4 py-3 text-slate-700 text-xs max-w-xs">
                                <span className="line-clamp-2">{e.string1}</span>
                              </td>
                              <td className="px-4 py-3 text-slate-700 text-xs max-w-xs">
                                <span className="line-clamp-2">{e.string2}</span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

        </motion.div>
      </div>
    </div>
  );
}
