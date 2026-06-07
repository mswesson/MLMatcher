/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, Check, X, Code, Terminal, Play, 
  Download, Zap, Layers, Cpu, Sparkles, Star, 
  Binary, ListChecks, ArrowUpDown, Flame, HelpCircle 
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { AVAILABLE_FEATURES, FeatureId, TrainingLog, TaskStatus, TrainingMetrics } from '../types';

export default function StepTraining() {
  // Config state
  const [selectedFeatures, setSelectedFeatures] = useState<FeatureId[]>(
    AVAILABLE_FEATURES.filter(f => f.defaultChecked).map(f => f.id)
  );
  
  // File upload states
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Training state
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<TrainingLog[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);

  // Terminal autoscroll helper
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  // Clean active SSE listeners
  const sseRef = useRef<EventSource | null>(null);
  useEffect(() => {
    return () => {
      if (sseRef.current) {
        sseRef.current.close();
      }
    };
  }, []);

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
      if (droppedFile.name.endsWith('.csv')) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError('Пожалуйста, загрузите только CSV-файл.');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const loadDemoDataset = () => {
    const csvContent = `Строка 1,Строка 2
ООО "Газпром инвест",ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "ГАЗПРОМ ИНВЕСТ"
ИП Иванов Иван Павлович,Иванов И. П.
Apple Inc. iPhone 15 Pro,iPhone 15 Pro Max 256GB Apple
Смартфон Samsung Galaxy S24,Samsung Galaxy S 24 Ultra
ИКЕА Домодедово Мебель,IKEA Domodedovo Mebel LLC
АО "Тинькофф Банк",Акционерное Общество "Т-Банк"
ООО Яндекс Такси,Yandex Taxi Service LLC
Nike Air Force 1 Sneakers,Кроссовки Nike Air Force 1 оригинал
Кофемашина DeLonghi Magnifica,Кофемашина Делонги Магнифика автомат
Шоколад Альпен Гольд молочный,Шоколад молочный Alpen Gold 85г`;

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const demoFile = new File([blob], 'ml_matcher_demo_dataset.csv', { type: 'text/csv' });
    setFile(demoFile);
    setError(null);
  };

  const toggleFeature = (id: FeatureId) => {
    if (selectedFeatures.includes(id)) {
      setSelectedFeatures(selectedFeatures.filter(fid => fid !== id));
    } else {
      setSelectedFeatures([...selectedFeatures, id]);
    }
  };

  const startTraining = async () => {
    if (!file) {
      setError('Загрузите файл CSV перед началом обучения.');
      return;
    }
    if (selectedFeatures.length === 0) {
      setError('Выберите хотя бы один признак (feature) для расчета.');
      return;
    }

    setError(null);
    setLogs([]);
    setMetrics(null);
    setTaskId(null);
    setProgress(0);
    setStatus('preparing');

    try {
      // Шаг 1: загрузить датасет и получить dataset_id
      const uploadForm = new FormData();
      uploadForm.append('file', file);
      const uploadRes = await fetch('/api/v1/dataset/upload', {
        method: 'POST',
        body: uploadForm,
      });
      if (!uploadRes.ok) {
        const errJson = await uploadRes.json();
        throw new Error(errJson.error || 'Ошибка загрузки датасета');
      }
      const { dataset_id } = await uploadRes.json();

      // Шаг 2: запустить обучение с dataset_id и выбранными фичами
      const trainRes = await fetch('/api/v1/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id, features: selectedFeatures }),
      });

      if (!trainRes.ok) {
        const errJson = await trainRes.json();
        throw new Error(errJson.error || 'Ошибка запуска обучения');
      }

      const data = await trainRes.json();
      const newTaskId = data.task_id;
      setTaskId(newTaskId);

      // Connect to Server-Sent Events (SSE) for training updates
      if (sseRef.current) {
        sseRef.current.close();
      }

      const sse = new EventSource(`/api/v1/training/status/${newTaskId}`);
      sseRef.current = sse;

      sse.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.error) {
            setError(payload.error);
            setStatus('failed');
            sse.close();
            return;
          }

          const { status: currentStatus, progress: currentProgress, log, metrics: eventMetrics } = payload;

          setStatus(currentStatus);
          setProgress(currentProgress);

          if (log) {
            setLogs(prev => [...prev, log]);
          }

          if (eventMetrics) {
            setMetrics(eventMetrics);
          }

          if (currentStatus === 'completed') {
            sse.close();
          } else if (currentStatus === 'failed') {
            setError('Произошел сбой во время обучения алгоритма CatBoost.');
            sse.close();
          }
        } catch (e) {
          console.error('SSE JSON error:', e);
        }
      };

      sse.onerror = (e) => {
        console.error('SSE Error event:', e);
        sse.close();
      };

    } catch (err: any) {
      setError(err.message || 'Ошибка подключения к серверу.');
      setStatus('failed');
    }
  };

  const downloadModel = () => {
    if (!taskId) return;
    window.location.href = `/api/v1/training/download/${taskId}`;
  };

  // Feature icon helpers
  const getFeatureIcon = (id: FeatureId) => {
    switch (id) {
      case 'levenshtein':
        return <span className="text-violet-500 font-semibold font-mono text-sm group-hover:scale-110 transition-transform">Ld</span>;
      case 'jaro_winkler':
        return <Star className="h-4 w-4 text-amber-500 group-hover:rotate-12 transition-transform" />;
      case 'dice':
        return <Layers className="h-4 w-4 text-blue-500 group-hover:scale-110 transition-transform" />;
      case 'number_match':
        return <Binary className="h-4 w-4 text-teal-500 group-hover:scale-110 transition-transform" />;
      case 'number_similarity':
        return <span className="text-teal-600 font-semibold font-mono text-sm group-hover:scale-110 transition-transform">≋</span>;
      case 'word_intersection':
        return <ListChecks className="h-4 w-4 text-emerald-500 group-hover:scale-110 transition-transform" />;
      case 'token_set_ratio':
        return <Zap className="h-4 w-4 text-emerald-500 group-hover:scale-110 transition-transform" />;
      case 'partial_ratio':
        return <Flame className="h-4 w-4 text-orange-500 group-hover:scale-110 transition-transform" />;
      case 'length_diff':
        return <ArrowUpDown className="h-4 w-4 text-rose-500 group-hover:translate-y-0.5 transition-transform" />;
      case 'tfidf_cosine':
        return <Sparkles className="h-4 w-4 text-indigo-500 group-hover:animate-pulse" />;
      case 'embedding_cosine':
        return <span className="text-fuchsia-500 font-semibold font-mono text-sm group-hover:scale-110 transition-transform">≈</span>;
      default:
        return <HelpCircle className="h-4 w-4" />;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
      
      {/* LEFT COLUMN - Information Box */}
      <div className="col-span-1 lg:col-span-5 flex flex-col justify-center py-6">
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="space-y-6"
        >
          <div className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-semibold text-blue-700 bg-blue-50 rounded-full border border-blue-100 uppercase tracking-wider">
            Шаг 1 • Обучение алгоритма
          </div>
          
          <h2 className="font-sans text-3xl font-extrabold tracking-tight text-slate-900 leading-tight sm:text-4xl">
            Обучайте модели <br />
            для точного сравнения строк
          </h2>
          
          <p className="font-sans text-base text-slate-500 leading-relaxed">
            Загрузите эталонный датасет, выберите математические признаки и запустите обучение. 
            Система сгенерирует сложные негативные примеры и обучит модель с помощью CatBoost.
          </p>

          <div className="space-y-5 pt-4 border-t border-slate-100">
            {/* Feature 1 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-blue-50 text-blue-600 border border-blue-100">
                <Zap className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">
                  Автоматическая генерация негативов
                </h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  На основе TF-IDF и косинусного сходства
                </p>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-indigo-50 text-indigo-600 border border-indigo-100">
                <Layers className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">
                  Гибкий выбор признаков
                </h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Комбинируйте математические метрики
                </p>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 flex items-center justify-center h-10 w-10 rounded-xl bg-emerald-50 text-emerald-600 border border-emerald-100">
                <Check className="h-5 w-5" />
              </div>
              <div>
                <h4 className="font-sans text-sm font-semibold text-slate-900">
                  Готовая модель в ZIP
                </h4>
                <p className="font-sans text-xs text-slate-400 mt-0.5">
                  Файлы <code className="font-mono text-[11px] bg-slate-100 px-1 py-0.5 rounded text-indigo-600">model.cbm</code> и <code className="font-mono text-[11px] bg-slate-100 px-1 py-0.5 rounded text-indigo-600">meta.json</code> в одном архиве
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* RIGHT COLUMN - Training Sandbox */}
      <div className="col-span-1 lg:col-span-7">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="relative bg-white/95 backdrop-blur-md rounded-3xl p-6 sm:p-8 shadow-2xl shadow-blue-500/5 border border-slate-100 space-y-8"
        >
          {/* Step 1.1: Upload Dataset */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-sans text-sm font-bold text-slate-900 tracking-tight uppercase flex items-center gap-2">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-white text-[10px] font-mono">1</span>
                Загрузите CSV датасет
              </h3>
              {!file && (
                <button
                  type="button"
                  id="btn-load-demo"
                  onClick={loadDemoDataset}
                  className="font-sans text-xs font-semibold text-blue-600 hover:text-blue-800 transition-colors cursor-pointer"
                >
                  Загрузить демо датасет
                </button>
              )}
            </div>

            <div
              id="file-drop-zone"
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`relative group flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-2xl cursor-pointer select-none transition-all duration-300 ${
                dragActive
                  ? 'border-indigo-500 bg-indigo-50/40 shadow-inner'
                  : file
                  ? 'border-emerald-200 bg-emerald-50/10'
                  : 'border-slate-200 hover:border-slate-350 hover:bg-slate-50/20'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />

              {file ? (
                <div className="flex w-full items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-emerald-100 text-emerald-700">
                      <Code className="h-5 w-5" />
                    </div>
                    <div>
                      <h4 className="font-sans text-sm font-semibold text-slate-800 break-all max-w-[200px] sm:max-w-md">
                        {file.name}
                      </h4>
                      <p className="font-mono text-xs text-slate-400 mt-0.5">
                        {(file.size / 1024).toFixed(1)} KB • Готов к обработке
                      </p>
                    </div>
                  </div>
                  <button
                    type="button"
                    title="Удалить файл"
                    onClick={(e) => {
                      e.stopPropagation();
                      setFile(null);
                      setError(null);
                    }}
                    className="flex h-8 w-8 items-center justify-center rounded-full text-slate-400 hover:bg-slate-100 hover:text-slate-800 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center text-center space-y-2">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-50 text-blue-600 border border-blue-100/30 group-hover:scale-105 transition-transform duration-300">
                    <Upload className="h-5 w-5" />
                  </div>
                  <div className="space-y-1">
                    <p className="font-sans text-sm font-semibold text-slate-700">
                      Перетащите файл сюда или <span className="text-blue-600 underline decoration-2">нажмите для выбора</span>
                    </p>
                    <p className="font-mono text-xs text-slate-400 uppercase tracking-widest">
                      CSV, до 50MB
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Step 1.2: Checkbox Features Constructors */}
          <div className="space-y-3">
            <h3 className="font-sans text-sm font-bold text-slate-900 tracking-tight uppercase flex items-center gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-900 text-white text-[10px] font-mono">2</span>
              Выберите признаки (features)
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3" id="features-constructor">
              {AVAILABLE_FEATURES.map((feature) => {
                const isSelected = selectedFeatures.includes(feature.id);
                return (
                  <div
                    id={`feature-card-${feature.id}`}
                    key={feature.id}
                    onClick={() => toggleFeature(feature.id)}
                    className={`group flex items-start gap-3 p-3.5 rounded-xl border cursor-pointer select-none transition-all duration-300 ${
                      isSelected
                        ? 'border-blue-200 bg-blue-50/20 shadow-sm shadow-blue-500/5'
                        : 'border-slate-150 bg-white hover:border-slate-300 hover:shadow-sm'
                    }`}
                  >
                    <div className="flex-shrink-0 flex items-center justify-center h-8 w-8 rounded-lg bg-slate-50 border border-slate-100">
                      {getFeatureIcon(feature.id)}
                    </div>
                    <div className="flex-1 min-w-0 pr-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-sans text-xs font-semibold text-slate-800 truncate leading-tight">
                          {feature.label}
                        </span>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => {}} // Box toggled via container onClick
                          className="h-3.5 w-3.5 rounded-md border-slate-300 text-blue-600 focus:ring-blue-500 flex-shrink-0 cursor-pointer pointer-events-none"
                        />
                      </div>
                      <p className="font-sans text-[11px] text-slate-400 leading-snug mt-1 group-hover:text-slate-500 transition-colors">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Action Trigger Block */}
          <div className="space-y-4 pt-4 border-t border-slate-100">
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start gap-2 p-3 text-xs bg-rose-50 border border-rose-100 text-rose-700 rounded-xl"
              >
                <X className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <span>{error}</span>
              </motion.div>
            )}

            {/* Simulated Training View or Trigger */}
            {status !== 'idle' && (
              <div className="space-y-3" id="training-progress-console">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-sans font-semibold text-indigo-700 flex items-center gap-1.5 capitalize">
                    {status === 'completed' ? (
                      <span className="inline-flex items-center gap-1.5 text-emerald-700">
                        <Check className="h-3.5 w-3.5" /> Обучение завершено
                      </span>
                    ) : status === 'failed' ? (
                      <span className="inline-flex items-center gap-1.5 text-rose-700">
                        <X className="h-3.5 w-3.5" /> Сбой обучения
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-2">
                        <Flame className="h-4.5 w-4.5 text-orange-500 animate-bounce" />
                        Обучение модели...
                      </span>
                    )}
                  </span>
                  <span className="font-mono font-bold text-slate-800">{progress}%</span>
                </div>

                {/* Progress bar */}
                <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      status === 'completed'
                        ? 'bg-emerald-500'
                        : status === 'failed'
                        ? 'bg-rose-500'
                        : 'bg-gradient-to-r from-blue-500 via-indigo-500 to-violet-500'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${progress}%` }}
                    transition={{ ease: 'easeOut' }}
                  />
                </div>

                {/* Terminal logging panel */}
                <div className="relative rounded-xl border border-slate-950/20 bg-slate-950 p-4 shadow-lg">
                  <div className="absolute top-2.5 right-3 flex items-center gap-1">
                    <Terminal className="h-3.5 w-3.5 text-slate-500" />
                    <span className="font-mono text-[9px] text-slate-500 uppercase">logs</span>
                  </div>
                  <div className="flex items-center gap-1.5 border-b border-white/5 pb-2 mb-2">
                    <span className="h-2 w-2 rounded-full bg-red-500" />
                    <span className="h-2 w-2 rounded-full bg-yellow-500" />
                    <span className="h-2 w-2 rounded-full bg-green-500" />
                    <span className="font-sans text-[10px] text-slate-400 font-semibold ml-1.5">
                      CatBoost Core Trainer Console
                    </span>
                  </div>

                  <div 
                    ref={terminalRef}
                    className="h-44 overflow-y-auto space-y-1.5 font-mono text-xs pr-2 text-white/95"
                    style={{ scrollBehavior: 'smooth' }}
                  >
                    {logs.map((logItem, index) => {
                      let colorClass = 'text-slate-400';
                      let prefix = '[INFO]';
                      if (logItem.type === 'success') {
                        colorClass = 'text-emerald-400';
                        prefix = '[SUCCESS]';
                      } else if (logItem.type === 'warn') {
                        colorClass = 'text-amber-400';
                        prefix = '[WARNING]';
                      } else if (logItem.type === 'iteration') {
                        colorClass = 'text-indigo-300';
                        prefix = '[TRAIN]';
                      }
                      
                      return (
                        <div key={index} className="flex gap-2">
                          <span className="text-slate-600 font-medium whitespace-nowrap">
                            {logItem.timestamp}
                          </span>
                          <span className={`font-semibold shrink-0 whitespace-nowrap ${colorClass}`}>
                            {prefix}
                          </span>
                          <span className="text-slate-100 breakdown-word font-normal">
                            {logItem.message}
                          </span>
                        </div>
                      );
                    })}
                    {status !== 'completed' && status !== 'failed' && (
                      <div className="flex items-center gap-1 text-slate-500 italic text-[11px] animate-pulse">
                        <span className="h-1.5 w-1.5 rounded-full bg-indigo-500 h-1.5 animate-bounce" />
                        Ожидание очередного лога от ядра CatBoost...
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {status === 'completed' && metrics && (() => {
              const errPct = metrics.error_rate * 100;
              const tone = errPct < 5
                ? { box: 'bg-emerald-50 border-emerald-200', accent: 'text-emerald-700', label: 'Модель обучилась хорошо' }
                : errPct < 15
                ? { box: 'bg-amber-50 border-amber-200', accent: 'text-amber-700', label: 'Качество среднее — есть что улучшить' }
                : { box: 'bg-rose-50 border-rose-200', accent: 'text-rose-700', label: 'Модель обучилась криво' };
              const thr = Math.round(metrics.match_threshold * 100);
              const cell = (title: string, value: string) => (
                <div className="text-center">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold">{title}</p>
                  <p className="text-base font-bold text-slate-700">{value}</p>
                </div>
              );
              return (
                <motion.div
                  key="metrics-section"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`mb-3 p-4 rounded-xl border ${tone.box}`}
                >
                  <div className="flex items-baseline justify-between mb-3">
                    <span className="text-xs font-semibold uppercase tracking-widest text-slate-500">
                      Тестирование на 30% датасета
                    </span>
                    <span className={`text-xs font-semibold ${tone.accent}`}>{tone.label}</span>
                  </div>
                  <div className="flex items-center gap-4 mb-4">
                    <div>
                      <p className={`text-3xl font-bold ${tone.accent}`}>{errPct.toFixed(1)}%</p>
                      <p className="text-[11px] text-slate-500 leading-tight mt-0.5">
                        ошибок — на стольких верных парах модель<br />дала уверенность ниже {thr}%
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-3 pt-3 border-t border-slate-200/70">
                    {cell(`Уверенных (≥${thr}%)`, `${(metrics.positive_recall_at_threshold * 100).toFixed(1)}%`)}
                    {cell('Специфичность', `${(metrics.negative_specificity_at_threshold * 100).toFixed(1)}%`)}
                    {cell('ROC-AUC', metrics.auc.toFixed(3))}
                    {cell('F1', metrics.f1.toFixed(3))}
                    {cell('Accuracy', `${(metrics.accuracy * 100).toFixed(1)}%`)}
                    {cell('Train / Test', `${metrics.train_size} / ${metrics.test_size}`)}
                  </div>
                </motion.div>
              );
            })()}

            <AnimatePresence mode="wait">
              {status === 'completed' ? (
                <motion.div
                  key="download-section"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col sm:flex-row gap-2"
                >
                  <button
                    type="button"
                    onClick={downloadModel}
                    className="flex-1 h-12 flex items-center justify-center gap-2 bg-gradient-to-r from-emerald-650 to-teal-500 hover:from-emerald-700 hover:to-teal-600 shadow-md shadow-emerald-500/10 text-white font-sans text-sm font-semibold rounded-xl cursor-pointer transition-all duration-300 active:scale-98"
                  >
                    <Download className="h-4 w-4" />
                    <span>Скачать модель в формате ZIP</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setStatus('idle');
                      setFile(null);
                      setMetrics(null);
                    }}
                    className="h-12 px-5 flex items-center justify-center bg-slate-100 hover:bg-slate-200 text-slate-600 font-sans text-sm font-semibold rounded-xl transition-all cursor-pointer"
                  >
                    Обучить заново
                  </button>
                </motion.div>
              ) : (
                <motion.button
                  key="train-trigger"
                  type="button"
                  id="btn-train-model"
                  onClick={startTraining}
                  disabled={status !== 'idle' && status !== 'failed'}
                  className={`w-full h-12 flex items-center justify-center gap-2 bg-gradient-to-r from-blue-600 via-indigo-500 to-violet-500 text-white font-sans text-sm font-semibold rounded-xl shadow-lg shadow-blue-500/15 cursor-pointer transition-all duration-300 hover:from-blue-700 hover:to-violet-650 ${
                    status !== 'idle' && status !== 'failed' ? 'opacity-50 cursor-not-allowed' : 'active:scale-98'
                  }`}
                >
                  {status !== 'idle' && status !== 'failed' ? (
                    <>
                      <Cpu className="h-4 w-4 animate-spin" />
                      <span>Вычисление весов CatBoost...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 fill-white" />
                      <span>Обучить модель</span>
                    </>
                  )}
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>

    </div>
  );
}
