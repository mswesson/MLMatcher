/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef } from 'react';
import { 
  Upload, Check, X, Sliders, Play, 
  ArrowUpDown, Zap, Shield, Lock, Trash2, 
  FileArchive, HelpCircle, Flame, CheckCircle2,
  AlertTriangle, RefreshCw
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { AVAILABLE_FEATURES, FeatureId, InferenceResult } from '../types';

export default function StepInference() {
  // Model state
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelMeta, setModelMeta] = useState<{ taskId: string; features: FeatureId[]; datasetName: string } | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Inputs state
  const [string1, setString1] = useState('');
  const [string2, setString2] = useState('');

  // Inference state
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.endsWith('.zip')) {
        processModelZip(droppedFile);
      } else {
        setError('Пожалуйста, загрузите только ZIP-архив модели, полученный на Шаге 1.');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processModelZip(e.target.files[0]);
    }
    // Сбрасываем value, чтобы повторный выбор того же файла снова вызвал onChange.
    e.target.value = '';
  };

  const processModelZip = async (zipFile: File) => {
    setModelFile(zipFile);
    setError(null);
    setResult(null);

    // We can parse the ZIP's meta.json by calling a lightweight analysis check or using client JSZip
    const JSZip = (await import('jszip')).default;
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const arrayBuffer = event.target?.result as ArrayBuffer;
        const zip = await JSZip.loadAsync(arrayBuffer);
        const metaFile = zip.file('meta.json');
        
        if (!metaFile) {
          throw new Error('Архив не содержит файл meta.json со спецификацией модели.');
        }

        const metaText = await metaFile.async('string');
        const meta = JSON.parse(metaText);

        if (!meta.features || !Array.isArray(meta.features)) {
          throw new Error('Файл meta.json поврежден или содержит невалидный список фич.');
        }

        setModelMeta({
          taskId: meta.task_id || 'imported_task',
          features: meta.features,
          datasetName: meta.dataset_name || 'Импортированный датасет',
        });
        setError(null);

      } catch (err: any) {
        setError(err.message || 'Ошибка чтения ZIP-архива. Убедитесь, что это корректная модель.');
        setModelFile(null);
        setModelMeta(null);
      }
    };
    reader.readAsArrayBuffer(zipFile);
  };

  const swapStrings = () => {
    const temp = string1;
    setString1(string2);
    setString2(temp);
  };

  const compareStrings = async () => {
    if (!modelFile) {
      setError('Загрузите ZIP-архив модели, полученный на Шаге 1.');
      return;
    }
    if (!string1.trim() || !string2.trim()) {
      setError('Введите обе строки для сравнения.');
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', modelFile);
      formData.append('string1', string1);
      formData.append('string2', string2);

      const response = await fetch('/api/v1/inference/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.error || 'Ошибка расчета прогноза');
      }

      const data = await response.json();
      setResult({
        matchProbability: data.match_probability,
        featuresUsed: data.features_used,
        featureValues: data.feature_values,
        timeMs: data.time_ms,
      });
    } catch (err: any) {
      setError(err.message || 'Ошибка выполнения инференса на сервере.');
    } finally {
      setLoading(false);
    }
  };

  const getFeatureDetailLabel = (id: FeatureId, score: number) => {
    switch (id) {
      case 'levenshtein':
        return `${Math.round(score * 100)}% сходства символов (редакционная метрика)`;
      case 'jaro_winkler':
        return `${Math.round(score * 100)}% Jaro-Winkler сходство`;
      case 'dice':
        return `${Math.round(score * 100)}% сходства по биграммам (Sorensen-Dice)`;
      case 'number_match':
        return score === 1.0 ? 'Цифры полностью совпадают' : score === 0.0 ? 'Цифры в строках не совпадают!' : 'Цифры отсутствуют в обеих строках';
      case 'number_similarity':
        return `${Math.round(score * 100)}% близости чисел (учитывает величину различия дозировок)`;
      case 'word_intersection':
        return `${Math.round(score * 100)}% слов пересекаются друг с другом`;
      case 'length_diff':
        return `Разница длин: ${score} симв. (меньше – лучше)`;
      case 'tfidf_cosine':
        return `${Math.round(score * 100)}% совпадения векторов символьных триграмм TF-IDF`;
      case 'embedding_cosine':
        return `${Math.round(score * 100)}% семантического сходства (мультиязычные эмбеддинги)`;
      default:
        return 'Оценка метрики завершена';
    }
  };

  const getFeatureTitle = (id: FeatureId) => {
    return AVAILABLE_FEATURES.find(f => f.id === id)?.label || id;
  };

  // Human explanation summary generator
  const getInterpretationText = (prob: number) => {
    if (prob >= 0.82) {
      return {
        title: 'Высокая степень совпадения!',
        description: 'Алгоритм CatBoost уверен, что эти строки описывают один и тот же объект. Выявлен высокий уровень совпадения токенов и структуры лексем.',
        style: 'text-emerald-800 bg-emerald-50 border-emerald-100',
        dot: 'bg-emerald-500'
      };
    } else if (prob >= 0.45) {
      return {
        title: 'Возможное частичное совпадение',
        description: 'Обнаружено совпадение по части ключевых слов или сходству символов, но присутствуют значительные расхождения. Рекомендуется ручная верификация.',
        style: 'text-amber-800 bg-amber-50 border-amber-100',
        dot: 'bg-amber-500'
      };
    } else {
      return {
        title: 'Строки не совпадают',
        description: 'Слишком низкое совпадение по всем математическим признакам. Модель определила данные объекты как независимые сущности.',
        style: 'text-rose-800 bg-rose-50 border-rose-100',
        dot: 'bg-rose-500'
      };
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
      
      {/* LEFT COLUMN - Information Box */}
      <div className="col-span-1 lg:col-span-11 xl:col-span-5 flex flex-col justify-center py-6">
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          <div className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold text-indigo-700 bg-indigo-50 rounded-full border border-indigo-100 uppercase tracking-wider">
            Шаг 2 • Точечный инференс
          </div>
          
          <h2 className="font-sans text-3xl font-extrabold tracking-tight text-slate-900 leading-tight sm:text-4xl">
            Проверяйте сходство <br />
            любых двух строк
          </h2>
          
          <p className="font-sans text-base text-slate-500 leading-relaxed">
            Загрузите обученную модель в формате ZIP (которая содержит конфигурацию с весами) и введите две строки. 
            Система рассчитает вероятность их совпадения на основе выбранных при обучении признаков.
          </p>

          <div className="space-y-5 pt-4 border-t border-slate-100">
            {/* Feature 1 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-orange-50 text-orange-600 border border-orange-100">
                <Zap className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">
                  Мгновенный результат
                </h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Быстрый расчет на основе обученной модели
                </p>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-blue-50 text-blue-600 border border-blue-100">
                <Shield className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">
                  Надёжность расчетов
                </h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Используются те же математические веса, что и при обучении
                </p>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-purple-50 text-purple-600 border border-purple-100">
                <Lock className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">
                  Конфиденциальность
                </h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Данные обрабатываются локально и не сохраняются
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* RIGHT COLUMN - Interrogation Sandbox */}
      <div className="col-span-1 lg:col-span-12 xl:col-span-7">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="bg-white/95 backdrop-blur-md rounded-3xl p-6 sm:p-8 shadow-2xl shadow-blue-500/5 border border-slate-100 space-y-7"
        >
          {/* Section 2.1: Model selection */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-sans text-sm font-bold text-slate-900 tracking-tight uppercase flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-white text-[10px] font-mono">1</span>
                Загрузите модель (ZIP)
              </h3>
            </div>

            <div
              id="model-zip-dropzone"
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => !modelFile && fileInputRef.current?.click()}
              className={`relative flex flex-col items-center justify-center p-5 border-2 border-dashed rounded-2xl transition-all duration-300 ${
                modelFile
                  ? 'border-emerald-250 bg-emerald-50/10 cursor-default'
                  : dragActive
                  ? 'border-indigo-550 bg-indigo-50/40 shadow-inner cursor-pointer'
                  : 'border-slate-200 hover:border-slate-350 hover:bg-slate-50/20 cursor-pointer'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".zip"
                onChange={handleFileChange}
                className="hidden"
                disabled={!!modelFile}
              />

              {modelFile ? (
                <div className="flex w-full items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-100 text-blue-700">
                      <FileArchive className="h-5 w-5" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <h4 className="font-sans text-sm font-semibold text-slate-800 truncate max-w-[180px] sm:max-w-xs">
                          {modelFile.name}
                        </h4>
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 whitespace-nowrap">
                          <CheckCircle2 className="h-3 w-3" /> Модель загружена
                        </span>
                      </div>
                      <p className="font-sans text-xs text-slate-400 mt-1">
                        Размер: {(modelFile.size / 1024).toFixed(1)} KB • Конфиг: {modelMeta?.features.length} фич
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    title="Удалить модель"
                    onClick={(e) => {
                      e.stopPropagation();
                      setModelFile(null);
                      setModelMeta(null);
                      setResult(null);
                      setError(null);
                      // Сбрасываем input, иначе повторная загрузка модели не сработает.
                      if (fileInputRef.current) fileInputRef.current.value = '';
                    }}
                    className="flex h-8 w-8 items-center justify-center rounded-full text-slate-400 hover:bg-slate-100 hover:text-red-650 transition-colors"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-50 text-blue-600 border border-blue-50/20">
                    <Upload className="h-5 w-5" />
                  </div>
                  <div className="space-y-1">
                    <p className="font-sans text-sm font-semibold text-slate-700">
                      Перетащите ZIP-файл модели сюда или <span className="text-blue-600 underline">нажмите для выбора</span>
                    </p>
                    <p className="font-sans text-[11px] text-slate-400">
                      Файл должен содержать: <code className="font-mono bg-slate-100 text-[10px] px-1 py-0.5 rounded">model.cbm</code> и <code className="font-mono bg-slate-100 text-[10px] px-1 py-0.5 rounded">meta.json</code>
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Section 2.2: Dual inputs */}
          <div className="space-y-3 relative">
            <h3 className="font-sans text-sm font-bold text-slate-900 tracking-tight uppercase flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-white text-[10px] font-mono">2</span>
              Введите строки для сравнения
            </h3>

            <div className="space-y-3 relative">
              {/* String 1 input */}
              <div className="space-y-1 relative" id="input-string-1">
                <label className="font-sans text-xs font-semibold text-slate-400 uppercase tracking-widest pl-1">
                  Строка 1
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={string1}
                    onChange={(e) => setString1(e.target.value)}
                    placeholder="Введите первую строку"
                    className="w-full h-11 px-4 pr-10 text-sm border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-sans shadow-sm transition-shadow"
                  />
                  <span className="absolute right-3.5 top-1/2 -translate-y-1/2 font-sans font-mono text-xs font-bold text-slate-350">
                    Aa
                  </span>
                </div>
              </div>

              {/* Graphical Interchange Link */}
              <div className="flex justify-center -my-1.5 relative z-10">
                <button
                  type="button"
                  id="btn-swap-strings"
                  onClick={swapStrings}
                  title="Поменять строки местами"
                  className="flex h-7 w-7 items-center justify-center bg-white hover:bg-slate-50 border border-slate-100 rounded-lg text-blue-600 shadow-md shadow-blue-500/5 cursor-pointer hover:scale-105 active:scale-95 transition-all"
                >
                  <ArrowUpDown className="h-3.5 w-3.5" />
                </button>
              </div>

              {/* String 2 input */}
              <div className="space-y-1 relative" id="input-string-2">
                <label className="font-sans text-xs font-semibold text-slate-400 uppercase tracking-widest pl-1">
                  Строка 2
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={string2}
                    onChange={(e) => setString2(e.target.value)}
                    placeholder="Введите вторую строку"
                    className="w-full h-11 px-4 pr-10 text-sm border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-sans shadow-sm transition-shadow"
                  />
                  <span className="absolute right-3.5 top-1/2 -translate-y-1/2 font-sans font-mono text-xs font-bold text-slate-350">
                    Aa
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Section 2.3: Results output */}
          <div className="space-y-4 pt-4 border-t border-slate-100">
            {error && (
              <div className="flex items-start gap-2 p-3 text-xs bg-rose-50 border border-rose-100 text-rose-700 rounded-xl">
                <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}

            {/* Comparer Trigger Action */}
            {!result ? (
              <button
                type="button"
                id="btn-compare-strings"
                onClick={compareStrings}
                disabled={loading}
                className={`w-full h-12 flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 via-indigo-500 to-violet-500 text-white font-sans text-sm font-semibold rounded-xl shadow-lg shadow-blue-500/15 cursor-pointer hover:from-blue-700 hover:to-violet-650 transition-all ${
                  loading ? 'opacity-70 cursor-not-allowed' : 'active:scale-98'
                }`}
              >
                {loading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    <span>Сравнение весов...</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-white" />
                    <span>Сравнить строки</span>
                  </>
                )}
              </button>
            ) : (
              <motion.div
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="space-y-5"
                id="inference-result-panel"
              >
                {/* Visual Circle Gauge & Summary */}
                <div className="grid grid-cols-1 sm:grid-cols-12 gap-5 items-center bg-slate-100/45 p-5 border border-slate-200/40 rounded-2xl">
                  {/* Circle gauge logic */}
                  <div className="col-span-1 sm:col-span-4 flex flex-col items-center justify-center">
                    <div className="relative flex items-center justify-center h-28 w-28">
                      {/* Ring path */}
                      <svg className="h-full w-full -rotate-90">
                        <circle
                          cx="56"
                          cy="56"
                          r="46"
                          className="text-slate-200"
                          strokeWidth="8"
                          stroke="currentColor"
                          fill="transparent"
                        />
                        <motion.circle
                          cx="56"
                          cy="56"
                          r="46"
                          strokeWidth="8"
                          stroke={
                            result.matchProbability >= 0.75
                              ? '#10b981' // emerald
                              : result.matchProbability >= 0.45
                              ? '#f59e0b' // amber
                              : '#ef4444' // rose
                          }
                          fill="transparent"
                          strokeDasharray={2 * Math.PI * 46}
                          initial={{ strokeDashoffset: 2 * Math.PI * 46 }}
                          animate={{ strokeDashoffset: (1 - result.matchProbability) * (2 * Math.PI * 46) }}
                          transition={{ duration: 0.8, ease: 'easeOut' }}
                          strokeLinecap="round"
                        />
                      </svg>
                      {/* Percentage inside ring */}
                      <div className="absolute flex flex-col items-center">
                        <span className="font-sans text-2xl font-black text-slate-800 tracking-tight leading-none">
                          {Math.round(result.matchProbability * 100)}%
                        </span>
                        <span className="font-sans text-[9px] text-slate-400 font-bold uppercase tracking-widest mt-0.5">
                          вероятность
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Quantitative Summary Text */}
                  <div className="col-span-1 sm:col-span-8 space-y-2">
                    {(() => {
                      const analysis = getInterpretationText(result.matchProbability);
                      return (
                        <div className={`p-4 rounded-xl border ${analysis.style} space-y-1`}>
                          <h4 className="font-sans text-[13px] font-bold tracking-tight flex items-center gap-1.5">
                            <span className={`h-2 w-2 rounded-full ${analysis.dot}`} />
                            {analysis.title}
                          </h4>
                          <p className="font-sans text-xs opacity-90 leading-normal">
                            {analysis.description}
                          </p>
                        </div>
                      );
                    })()}
                    <div className="flex items-center justify-between font-mono text-[10px] text-slate-400 px-1 pt-0.5">
                      <span>Время вычислений: {result.timeMs} мс</span>
                      <span>Фичи в ядре: {result.featuresUsed.length} единиц</span>
                    </div>
                  </div>
                </div>

                {/* Mathematical features breakdown details */}
                <div className="space-y-2.5">
                  <h4 className="font-sans text-xs font-bold text-slate-400 uppercase tracking-widest pl-1">
                    Детальная метрическая оценка (Feature Values)
                  </h4>
                  <div className="space-y-2">
                    {result.featuresUsed.map((fid) => {
                      const score = result.featureValues[fid] ?? 0;
                      const isLengthDiff = fid === 'length_diff';
                      // Приводим к единой шкале «качества» 0..1: больше — лучше.
                      // Для length_diff меньшая разница = лучше, поэтому инвертируем
                      // (0 симв. → 1.0, ≥20 симв. → 0). Цвет и заполнение — как у всех.
                      const goodness = isLengthDiff ? Math.max(0, 1 - score / 20) : score;
                      const barPercentage = goodness * 100;

                      return (
                        <div key={fid} className="p-3 bg-white border border-slate-150 rounded-xl space-y-1.5 shadow-sm">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-sans font-medium text-slate-800">
                              {getFeatureTitle(fid)}
                            </span>
                            <span className="font-mono font-bold text-slate-900 bg-slate-100 px-1.5 py-0.5 rounded text-[11px]">
                              {isLengthDiff ? `${score} симв.` : score.toFixed(3)}
                            </span>
                          </div>

                          {/* Mini Progress bar */}
                          <div className="h-1.5 w-full bg-slate-50 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full ${
                                goodness >= 0.75
                                  ? 'bg-emerald-500'
                                  : goodness >= 0.4
                                  ? 'bg-amber-400'
                                  : 'bg-slate-400'
                              }`}
                              style={{ width: `${barPercentage}%` }}
                            />
                          </div>

                          <p className="font-sans text-[11px] text-slate-450">
                            {getFeatureDetailLabel(fid, score)}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Compare Another Block */}
                <button
                  type="button"
                  onClick={() => setResult(null)}
                  className="w-full h-11 flex items-center justify-center gap-1.5 bg-slate-100 hover:bg-slate-200 text-slate-600 font-sans text-xs font-semibold rounded-xl cursor-pointer transition-colors"
                >
                  Сравнить другие строки
                </button>
              </motion.div>
            )}
          </div>
        </motion.div>
      </div>

    </div>
  );
}
