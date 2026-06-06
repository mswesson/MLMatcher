/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  BookOpen, Terminal, Code, Check, 
  ArrowRight, FileSpreadsheet, Sliders, 
  Download, Search, Play, Settings,
  Database, HelpCircle, ChevronRight, Copy,
  RefreshCw
} from 'lucide-react';
import { motion } from 'motion/react';
import { AVAILABLE_FEATURES } from '../types';

export default function Documentation() {
  const [activeSection, setActiveSection] = useState<'intro' | 'step1' | 'step2' | 'features' | 'api' | 'faq'>('intro');
  const [copiedText, setCopiedText] = useState<string | null>(null);

  // Playground stats
  const [activeApiRoute, setActiveApiRoute] = useState<string>('training_start');
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testingEndpoint, setTestingEndpoint] = useState(false);

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(label);
    setTimeout(() => setCopiedText(null), 1500);
  };

  const getApiDetails = (route: string) => {
    switch (route) {
      case 'training_start':
        return {
          method: 'POST',
          url: '/api/v1/training/start',
          desc: 'Инициация асинхронного процесса обучения.',
          reqBody: 'multipart/form-data\n  file: File (CSV-файл)\n  features: string[] (Массив идентификаторов признаков)',
          resBody: '{\n  "task_id": "task_xyz123"\n}',
          curl: `curl -X POST "${window.location.protocol}//${window.location.host}/api/v1/training/start" \\\n  -F "file=@/path/to/dataset.csv" \\\n  -F "features=[\\"levenshtein\\",\\"jaro_winkler\\"]"`,
          testResponse: '{\n  "task_id": "task_interactive_test_909"\n}'
        };
      case 'training_status':
        return {
          method: 'GET',
          url: '/api/v1/training/status/{task_id}',
          desc: 'Стриминг логов обучения в режиме реального времени через SSE (Server-Sent Events).',
          reqBody: 'Параметры пути:\n  task_id: string (идентификатор запущенной задачи)',
          resBody: 'text/event-stream\n\nПример кадра данных:\ndata: {"taskId": "task_xyz", "status": "training", "progress": 68, "log": {"timestamp": "12:35", "type": "iteration", "message": "[CatBoost] Iteration 20: loss=0.435, accuracy=0.81"}}',
          curl: `curl -N "${window.location.protocol}//${window.location.host}/api/v1/training/status/task_xyz123"`,
          testResponse: 'data: {"taskId": "task_demo", "status": "completed", "progress": 100, "log": {"type": "success", "message": "Обучение завершено"}}'
        };
      case 'training_download':
        return {
          method: 'GET',
          url: '/api/v1/training/download/{task_id}',
          desc: 'Скачивание готового ZIP-архива, содержащего веса обученной модели (model.cbm) и метаданные выбранных признаков (meta.json).',
          reqBody: 'Параметры пути:\n  task_id: string (идентификатор запущенной задачи)',
          resBody: 'Binary file (application/zip)',
          curl: `curl -O -J "${window.location.protocol}//${window.location.host}/api/v1/training/download/task_xyz123"`,
          testResponse: '二进制流 / ZIP-архив download_stream'
        };
      case 'inference_predict':
        return {
          method: 'POST',
          url: '/api/v1/inference/predict',
          desc: 'Точечное сопоставление (матчинг) двух произвольных текстовых строк на основе весов и признаков загруженной модели.',
          reqBody: 'multipart/form-data\n  file: File (ZIP-архив модели)\n  string1: string (Первая строка)\n  string2: string (Вторая строка)',
          resBody: '{\n  "match_probability": 0.942,\n  "features_used": ["levenshtein", "jaro_winkler", "word_intersection"],\n  "feature_values": {\n    "levenshtein": 0.88,\n    "jaro_winkler": 0.92,\n    "word_intersection": 1.0\n  },\n  "time_ms": 14\n}',
          curl: `curl -X POST "${window.location.protocol}//${window.location.host}/api/v1/inference/predict" \\\n  -F "file=@/path/to/model.zip" \\\n  -F "string1=Дюны отель" \\\n  -F "string2=Отель Дюны СПА"`,
          testResponse: '{\n  "match_probability": 0.881,\n  "features_used": [\n    "levenshtein",\n    "jaro_winkler"\n  ],\n  "feature_values": {\n    "levenshtein": 0.80,\n    "jaro_winkler": 0.89\n  },\n  "time_ms": 11\n}'
        };
      default:
        return null;
    }
  };

  const testApiRoute = async (route: string) => {
    setTestingEndpoint(true);
    setTestResult(null);

    // Simulate API ping or trigger real mock endpoint matching selections
    setTimeout(() => {
      const details = getApiDetails(route);
      if (details) {
        setTestResult(details.testResponse);
      }
      setTestingEndpoint(false);
    }, 600);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      
      {/* 1. LEFT SIDE NAVIGATION */}
      <div className="col-span-1 lg:col-span-3 bg-white/60 rounded-2xl p-4 border border-slate-100 flex flex-col gap-5 sticky top-20">
        <div>
          <span className="font-sans text-[11px] font-bold text-slate-400 uppercase tracking-widest pl-3 block mb-2">
            Содержание
          </span>
          <div className="space-y-1">
            <button
              onClick={() => setActiveSection('intro')}
              className={`w-full flex items-center justify-between px-3 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeSection === 'intro' ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span>Введение и архитектура</span>
              <ChevronRight className="h-3 w-3 opacity-60" />
            </button>
            <button
              onClick={() => setActiveSection('step1')}
              className={`w-full flex items-center justify-between px-3 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeSection === 'step1' ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span>Обучение модели (Шаг 1)</span>
              <ChevronRight className="h-3 w-3 opacity-60" />
            </button>
            <button
              onClick={() => setActiveSection('step2')}
              className={`w-full flex items-center justify-between px-3 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeSection === 'step2' ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span>Тестирование (Шаг 2)</span>
              <ChevronRight className="h-3 w-3 opacity-60" />
            </button>
            <button
              onClick={() => setActiveSection('features')}
              className={`w-full flex items-center justify-between px-3 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeSection === 'features' ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span>Математические признаки</span>
              <ChevronRight className="h-3 w-3 opacity-60" />
            </button>
            <button
              onClick={() => setActiveSection('api')}
              className={`w-full flex items-center justify-between px-3 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeSection === 'api' ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span>API Reference</span>
              <ChevronRight className="h-3 w-3 opacity-60" />
            </button>
            <button
              onClick={() => setActiveSection('faq')}
              className={`w-full flex items-center justify-between px-3 py-2 text-xs font-semibold rounded-lg transition-colors cursor-pointer ${
                activeSection === 'faq' ? 'bg-blue-50 text-blue-700' : 'text-slate-600 hover:bg-slate-50'
              }`}
            >
              <span>Частые вопросы (FAQ)</span>
              <ChevronRight className="h-3 w-3 opacity-60" />
            </button>
          </div>
        </div>

        {/* Info panel */}
        <div className="bg-slate-50 rounded-xl p-3.5 border border-slate-150">
          <h4 className="font-sans text-[11px] font-bold text-slate-800 uppercase tracking-widest flex items-center gap-1.5">
            <Settings className="h-3.5 w-3.5 text-blue-600 animate-spin" style={{ animationDuration: '6s' }} />
            ML Core: CatBoost
          </h4>
          <p className="font-sans text-[11px] text-slate-400 mt-1 lines-normal">
            Используется классификатор деревьев CatBoost для максимизации качества сопоставлений на гетерогенных текстовых признаках.
          </p>
        </div>
      </div>

      {/* 2. RIGHT SIDE CONTENT SPACE */}
      <div className="col-span-1 lg:col-span-9 bg-white/95 backdrop-blur-md rounded-3xl p-6 sm:p-8 shadow-xl shadow-blue-500/5 border border-slate-100 min-h-[500px]">
        <div className="relative">
          
          {/* Section Introduction */}
          {activeSection === 'intro' && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <div className="flex items-center gap-3 border-b border-slate-100 pb-4">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-550/10 text-blue-600">
                  <BookOpen className="h-5 w-5" />
                </div>
                <div>
                  <h2 className="font-sans text-xl font-bold text-slate-900 tracking-tight">Документация сервиса</h2>
                  <p className="font-sans text-xs text-slate-400">Техническое руководство по платформе ML-матчинга строк.</p>
                </div>
              </div>

              <div id="intro-doc-content" className="space-y-6">
                <div>
                  <h3 className="font-sans text-base font-bold text-slate-800">О сервисе ML Matcher</h3>
                  <p className="font-sans text-sm text-slate-500 mt-2 leading-relaxed">
                    Веб-сервис разработан для автоматизированного сопоставления (матчинга) двух текстовых строк с использованием классического деревьев принятия решений CatBoost и статистического анализа TF-IDF. 
                    Интерфейс спроектирован в виде пошагового мастера, разделяющего этапы настройки обучающей выборки и проверки реальных запросов.
                  </p>
                </div>

                {/* Быстрый старт */}
                <div className="space-y-4 pt-2">
                  <h4 className="font-sans text-xs font-bold text-slate-400 uppercase tracking-widest">
                    Процедура быстрой конфигурации («Быстрый старт»)
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
                    
                    {/* Step 1 */}
                    <div className="p-4 bg-slate-100/40 rounded-xl border border-slate-150 space-y-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-700 text-xs font-bold font-mono">1</span>
                      <h5 className="font-sans text-xs font-bold text-slate-800 flex items-center gap-1.5">
                        <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-600" />
                        Датасет CSV
                      </h5>
                      <p className="font-sans text-[11px] text-slate-450 leading-normal">
                        Подготовьте CSV с колонками «Строка 1» и «Строка 2», представляющие истинные матчи.
                      </p>
                    </div>

                    {/* Step 2 */}
                    <div className="p-4 bg-slate-100/40 rounded-xl border border-slate-150 space-y-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-700 text-xs font-bold font-mono">2</span>
                      <h5 className="font-sans text-xs font-bold text-slate-800 flex items-center gap-1.5">
                        <Sliders className="h-3.5 w-3.5 text-blue-600" />
                        Параметры фич
                      </h5>
                      <p className="font-sans text-[11px] text-slate-450 leading-normal">
                        Выберите подходящие математические признаки и запустите асинхронное обучение.
                      </p>
                    </div>

                    {/* Step 3 */}
                    <div className="p-4 bg-slate-100/40 rounded-xl border border-slate-150 space-y-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-700 text-xs font-bold font-mono">3</span>
                      <h5 className="font-sans text-xs font-bold text-slate-800 flex items-center gap-1.5">
                        <Download className="h-3.5 w-3.5 text-purple-650" />
                        Экспорт в ZIP
                      </h5>
                      <p className="font-sans text-[11px] text-slate-450 leading-normal">
                        Скачайте заархивированные веса CatBoost (.cbm) и конфигурацию признаков в meta.json.
                      </p>
                    </div>

                    {/* Step 4 */}
                    <div className="p-4 bg-slate-100/40 rounded-xl border border-slate-150 space-y-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-100 text-blue-700 text-xs font-bold font-mono">4</span>
                      <h5 className="font-sans text-xs font-bold text-slate-800 flex items-center gap-1.5">
                        <Search className="h-3.5 w-3.5 text-amber-600" />
                        Инференс
                      </h5>
                      <p className="font-sans text-[11px] text-slate-450 leading-normal">
                        Импортируйте ZIP модель и начните точечное сравнение произвольных строк.
                      </p>
                    </div>

                  </div>
                </div>

                {/* Архитектура */}
                <div className="p-4 bg-blue-50/15 border border-blue-100/40 rounded-2xl space-y-3">
                  <h4 className="font-sans text-sm font-bold text-blue-800 flex items-center gap-2">
                    <Database className="h-4 w-4" /> Архитектурный паттерн: Feature-based Virtual Slices
                  </h4>
                  <p className="font-sans text-xs text-slate-500 leading-relaxed">
                    Бэкенд спроектирован на FastAPI с разделением по бизнес-доменам. Сложные расчеты признаков полностью хардкодятся на сервере. Обучение модели выведено в фоновые процессы (BackgroundTasks) с передачей статуса клиентам по протоколу Server-Sent Events (SSE). 
                    Для предотвращения сбоев все состояния запущенных задач кэшируются в in-memory dictionary.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Section Step 1 */}
          {activeSection === 'step1' && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <h3 className="font-sans text-lg font-bold text-slate-900 border-b border-slate-100 pb-3">
                Обучение модели (Шаг 1)
              </h3>
              
              <div className="space-y-4 font-sans text-sm text-slate-600 leading-relaxed">
                <div>
                  <h4 className="font-bold text-slate-800">Как это работает:</h4>
                  <p className="mt-1">
                    После отправки файла CSV, оркестратор загружает датасет и приступает к формированию расширенной тренировочной выборки, содержащей как позитивные пары (Класс 1), так и негативные (Класс 0).
                  </p>
                </div>

                <div className="border-l-4 border-blue-100 pl-4 py-1">
                  <h4 className="font-bold text-slate-800">Процедура построения Hard Negatives (Класс 0):</h4>
                  <p className="mt-1 text-xs">
                    Для эффективного обучения модели нужны примеры строк, которые визуально или лексически очень похожи, но описывают разные объекты.
                    Бэкенд строит временный TF-IDF индекс по массиву всех элементов «Строка 2». Для каждого элемента «Строка 1» с помощью косинусного сходства вычисляются наиболее близкие кандидаты из «Строки 2» (исключая исходную правильную пару). Эти схожие строки маркируются как негативные образцы.
                  </p>
                </div>

                <div>
                  <h4 className="font-bold text-slate-800">Получаемые артефакты (ZIP архив):</h4>
                  <p className="mt-1">
                    По завершении оптимизационных проходов CatBoost компилирует архив, включающий:
                  </p>
                  <ul className="list-disc pl-5 mt-2 space-y-1 text-xs">
                    <li><code className="font-mono bg-slate-100 px-1 py-0.5 rounded text-indigo-600">model.cbm</code> – Двоичные скомпилированные веса дерева решений CatBoost, готовые для быстрой загрузки.</li>
                    <li><code className="font-mono bg-slate-100 px-1 py-0.5 rounded text-indigo-600">meta.json</code> – Конфигурационный маркер, определяющий список математических метрик (features), необходимых для восстановления признаков при инференсе.</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          )}

          {/* Section Step 2 */}
          {activeSection === 'step2' && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <h3 className="font-sans text-lg font-bold text-slate-900 border-b border-slate-100 pb-3">
                Тестирование и инференс (Шаг 2)
              </h3>
              
              <div className="space-y-4 font-sans text-sm text-slate-600 leading-relaxed">
                <div>
                  <h4 className="font-bold text-slate-800">Регламент распаковки и чтения модели:</h4>
                  <p className="mt-1">
                    На шаге проверки обученной модели клиент загружает ZIP-архив обратно в систему. Бэкенд распаковывает архив, извлекает конфигурационный маркер и загружает CatBoost веса в оперативную память.
                  </p>
                </div>

                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/50">
                  <h4 className="font-bold text-slate-800 text-xs">Маршрутизация признаков:</h4>
                  <p className="mt-1 text-xs leading-relaxed text-slate-500">
                    Оркестратор обращается к извлеченному <code className="font-mono text-[11px] bg-white px-1 py-0.5 border border-slate-150 rounded">meta.json</code>, чтобы понять, какие именно функции оценки нужно вычислить для двух присланных на сравнение текстовых строк. 
                    Если какая-то метрика не использовалась при обучении модели, расчет для неё пропускается ради оптимизации CPU. Полученный вектор числовых метрик передается в инференс-ядро CatBoost, возвращающее вероятность принадлежности пары к совпадающему Классу 1.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {/* Section Matematika */}
          {activeSection === 'features' && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <h3 className="font-sans text-lg font-bold text-slate-900 border-b border-slate-100 pb-3">
                Математические признаки сопоставления
              </h3>
              
              <div className="space-y-4">
                <p className="font-sans text-sm text-slate-600 leading-relaxed">
                  Система содержит встроенную библиотеку формул текстового сходства, вычисляемых на бэкенде. Пользователи могут комбинировать их в конструкторе для обучения модели:
                </p>

                <div className="space-y-3">
                  {AVAILABLE_FEATURES.map((feature) => (
                    <div key={feature.id} className="p-3 bg-slate-50 border border-slate-150 rounded-xl">
                      <h4 className="font-sans text-xs font-bold text-slate-800">{feature.label}</h4>
                      <p className="font-sans text-[11px] text-slate-500 mt-1 lines-normal">{feature.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* Section API Reference */}
          {activeSection === 'api' && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
              id="api-ref-panel"
            >
              <h3 className="font-sans text-lg font-bold text-slate-900 border-b border-slate-100 pb-3">
                Архитектурное описание API (REST Reference)
              </h3>

              {/* Endpoint selection tabs */}
              <div className="flex flex-wrap gap-1.5 bg-slate-100/60 p-1 rounded-xl">
                <button
                  onClick={() => { setActiveApiRoute('training_start'); setTestResult(null); }}
                  className={`px-3 py-1.5 text-[11px] font-bold rounded-lg cursor-pointer transition-colors ${
                    activeApiRoute === 'training_start' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-850'
                  }`}
                >
                  POST /training/start
                </button>
                <button
                  onClick={() => { setActiveApiRoute('training_status'); setTestResult(null); }}
                  className={`px-3 py-1.5 text-[11px] font-bold rounded-lg cursor-pointer transition-colors ${
                    activeApiRoute === 'training_status' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-850'
                  }`}
                >
                  GET /training/status
                </button>
                <button
                  onClick={() => { setActiveApiRoute('training_download'); setTestResult(null); }}
                  className={`px-3 py-1.5 text-[11px] font-bold rounded-lg cursor-pointer transition-colors ${
                    activeApiRoute === 'training_download' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-850'
                  }`}
                >
                  GET /training/download
                </button>
                <button
                  onClick={() => { setActiveApiRoute('inference_predict'); setTestResult(null); }}
                  className={`px-3 py-1.5 text-[11px] font-bold rounded-lg cursor-pointer transition-colors ${
                    activeApiRoute === 'inference_predict' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-slate-850'
                  }`}
                >
                  POST /inference/predict
                </button>
              </div>

              {(() => {
                const details = getApiDetails(activeApiRoute);
                if (!details) return null;
                return (
                  <div className="space-y-4 border border-slate-150 p-5 rounded-2xl relative bg-slate-50/20" id="api-interactive-details">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-1 text-[10px] font-black rounded-lg text-white ${
                          details.method === 'POST' ? 'bg-emerald-600' : 'bg-blue-600'
                        }`}>
                          {details.method}
                        </span>
                        <code className="font-mono text-sm font-bold text-slate-800">
                          {details.url}
                        </code>
                      </div>
                      
                      {/* Copy Curl Button */}
                      <button
                        type="button"
                        onClick={() => copyToClipboard(details.curl, activeApiRoute)}
                        className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-500 hover:text-slate-850 cursor-pointer"
                      >
                        {copiedText === activeApiRoute ? (
                          <>
                            <Check className="h-3.5 w-3.5 text-emerald-600" />
                            <span className="text-emerald-600">Скопировано!</span>
                          </>
                        ) : (
                          <>
                            <Copy className="h-3.5 w-3.5" />
                            <span>Скопировать cURL</span>
                          </>
                        )}
                      </button>
                    </div>

                    <p className="font-sans text-xs text-slate-500">
                      {details.desc}
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Left: Input Payload details */}
                      <div className="space-y-1.5">
                        <h5 className="font-sans text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                          Параметры запроса (Request Headers/Body)
                        </h5>
                        <pre className="p-3 bg-slate-950 text-slate-100 rounded-xl text-xs font-mono overflow-auto max-h-36">
                          {details.reqBody}
                        </pre>
                      </div>

                      {/* Right: Response body example */}
                      <div className="space-y-1.5">
                        <h5 className="font-sans text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                          Тело ответа (Expected JSON / Response)
                        </h5>
                        <pre className="p-3 bg-slate-950 text-slate-100 rounded-xl text-xs font-mono overflow-auto max-h-36">
                          {details.resBody}
                        </pre>
                      </div>
                    </div>

                    {/* Integrated sandbox tester */}
                    <div className="pt-3 border-t border-slate-150 flex flex-col gap-3">
                      <div className="flex items-center justify-between">
                        <span className="font-sans text-[11px] font-semibold text-slate-400">
                          Демонстрационный интерактивный тестер маршрута:
                        </span>
                        
                        <button
                          type="button"
                          onClick={() => testApiRoute(activeApiRoute)}
                          disabled={testingEndpoint}
                          className="px-3.5 py-1.5 flex items-center gap-1.5 bg-blue-600 hover:bg-blue-700 text-white font-sans text-xs font-bold rounded-lg cursor-pointer transition-colors active:scale-98"
                        >
                          {testingEndpoint ? (
                            <>
                              <RefreshCw className="h-3 w-3 animate-spin" />
                              <span>Вызов...</span>
                            </>
                          ) : (
                            <>
                              <Play className="h-3 w-3 fill-white" />
                              <span>Тестировать эндпоинт</span>
                            </>
                          )}
                        </button>
                      </div>

                      {testResult && (
                        <motion.div
                          initial={{ opacity: 0, y: 5 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="space-y-1.5"
                        >
                          <span className="font-sans text-[9px] font-bold text-slate-400 uppercase tracking-widest">
                            Фактический ответ песочницы:
                          </span>
                          <pre className="p-4 bg-slate-900 text-emerald-400 rounded-xl text-xs font-mono overflow-auto border border-emerald-500/10">
                            {testResult}
                          </pre>
                        </motion.div>
                      )}
                    </div>

                  </div>
                );
              })()}
            </motion.div>
          )}

          {/* Section FAQ */}
          {activeSection === 'faq' && (
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              <h3 className="font-sans text-lg font-bold text-slate-900 border-b border-slate-100 pb-3">
                Часто задаваемые вопросы (FAQ)
              </h3>
              
              <div className="space-y-4 font-sans text-sm text-slate-600 leading-relaxed">
                <div className="space-y-1">
                  <h4 className="font-bold text-slate-800 flex items-center gap-1.5">
                    <HelpCircle className="h-4 w-4 text-blue-600" />
                    Какой размер CSV файла поддерживает веб-сервис?
                  </h4>
                  <p className="text-slate-500 pl-5 text-xs">
                    По умолчанию ограничение на загрузку в FastAPI установлено в 100MB, а веб-интерфейс отображает рекомендацию до 50MB. Этого достаточно для обработки сотен тысяч пар строк.
                  </p>
                </div>

                <div className="space-y-1">
                  <h4 className="font-bold text-slate-800 flex items-center gap-1.5">
                    <HelpCircle className="h-4 w-4 text-blue-600" />
                    Где хранится обученная модель после тренировки?
                  </h4>
                  <p className="text-slate-500 pl-5 text-xs">
                    Веса модели упаковываются в ZIP-архив и скачиваются на устройство пользователя. На бэкенде задачи хранятся в кэше к оперативной памяти ограниченное время. Никакие персональные данные или файлы на сервере перманентно не складируются.
                  </p>
                </div>

                <div className="space-y-1">
                  <h4 className="font-bold text-slate-800 flex items-center gap-1.5">
                    <HelpCircle className="h-4 w-4 text-blue-600" />
                    Заменяет ли этот сервис полноценную интеграцию ML-сервиса?
                  </h4>
                  <p className="text-slate-500 pl-5 text-xs">
                    Да! Выгружаемый ZIP-архив модели содержит полностью рабочие конфигурации. Вы можете использовать полученный весовой файл <code className="font-mono bg-slate-100 px-1 py-0.5 rounded text-indigo-600">model.cbm</code> в любом Python / C++ коде, загружая модель стандартным методом <code className="font-mono">model.load_model("model.cbm")</code> библиотеки CatBoost.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

        </div>
      </div>

    </div>
  );
}
