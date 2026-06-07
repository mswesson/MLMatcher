/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type FeatureId =
  | 'levenshtein'
  | 'levenshtein_token_sort'
  | 'jaro_winkler'
  | 'dice'
  | 'number_match'
  | 'number_similarity'
  | 'word_intersection'
  | 'token_set_ratio'
  | 'partial_ratio'
  | 'length_diff'
  | 'tfidf_cosine'
  | 'embedding_cosine'
  | 'number_unit_match';

export interface FeatureConfig {
  id: FeatureId;
  label: string;
  description: string;
  defaultChecked: boolean;
  category: 'edit' | 'token' | 'sequence' | 'semantic';
}

export const AVAILABLE_FEATURES: FeatureConfig[] = [
  {
    id: 'levenshtein',
    label: 'Расстояние Левенштейна',
    description: 'Метрика редакционного расстояния между последовательностями символов.',
    defaultChecked: true,
    category: 'edit',
  },
  {
    id: 'levenshtein_token_sort',
    label: 'Левенштейн без учёта порядка',
    description: 'Расстояние Левенштейна после сортировки слов: порядок слов не влияет.',
    defaultChecked: false,
    category: 'edit',
  },
  {
    id: 'jaro_winkler',
    label: 'Сходство Джаро-Винклера',
    description: 'Оценивает символьное сходство с приоритетом совпадения начальных символов.',
    defaultChecked: true,
    category: 'edit',
  },
  {
    id: 'dice',
    label: 'Коэффициент Дайс (n-граммы)',
    description: 'Сходство на основе пересечения символьных биграмм строк.',
    defaultChecked: false,
    category: 'sequence',
  },
  {
    id: 'number_match',
    label: 'Совпадение цифр',
    description: 'Бинарный признак: 1 - если все цифры в строках совпадают, иначе 0.',
    defaultChecked: true,
    category: 'token',
  },
  {
    id: 'number_similarity',
    label: 'Близость чисел (градуированная)',
    description: 'Учитывает величину различия чисел: 4.5≈4 — высокий балл, 500 vs 250 — низкий. Чтобы числа весили больше.',
    defaultChecked: true,
    category: 'token',
  },
  {
    id: 'word_intersection',
    label: 'Длина пересечения слов',
    description: 'Количество общих слов (лексем) после токенизации и очистки.',
    defaultChecked: true,
    category: 'token',
  },
  {
    id: 'token_set_ratio',
    label: 'Token-set ratio',
    description: 'Сходство по множествам слов: устойчиво к перестановке и подмножеству слов (часть названия внутри другого с лишним текстом).',
    defaultChecked: false,
    category: 'token',
  },
  {
    id: 'partial_ratio',
    label: 'Partial ratio',
    description: 'Лучший частичный матч: одна строка целиком входит в другую с добавочным текстом.',
    defaultChecked: false,
    category: 'sequence',
  },
  {
    id: 'length_diff',
    label: 'Абсолютная разница длин',
    description: 'Разность длин сравниваемых строк по количеству символов.',
    defaultChecked: false,
    category: 'sequence',
  },
  {
    id: 'tfidf_cosine',
    label: 'Косинусное сходство TF-IDF',
    description: 'Вычисление близости векторов частотности символьных 3-грамм.',
    defaultChecked: true,
    category: 'sequence',
  },
  {
    id: 'embedding_cosine',
    label: 'Семантическое сходство (эмбеддинги)',
    description: 'Косинус мультиязычных эмбеддингов: ловит смысл, синонимы и разные алфавиты. Считается только на сервере.',
    defaultChecked: true,
    category: 'semantic',
  },
  {
    id: 'number_unit_match',
    label: 'Число + единица измерения',
    description: 'Сравнивает пары (число, единица): «50 см» ≠ «50 м», «14 г» ≠ «14 шт». Отличает одно и то же число в разных единицах.',
    defaultChecked: true,
    category: 'token',
  },
];

export type TaskStatus = 'idle' | 'preparing' | 'tfidf' | 'negatives' | 'training' | 'completed' | 'failed';

export interface TrainingLog {
  timestamp: string;
  type: 'info' | 'warn' | 'success' | 'iteration';
  message: string;
  progress: number;
}

// Метрики качества модели на honest test (30% датасета).
export interface TrainingMetrics {
  error_rate: number;                          // доля верных пар с proba < порога (цель < 0.05)
  positive_recall_at_threshold: number;        // доля уверенных совпадений (цель ≥ 0.95)
  negative_specificity_at_threshold: number;   // доля негативов ниже порога (защита от вырождения)
  match_threshold: number;                     // порог уверенного совпадения (напр. 0.9)
  auc: number;
  f1: number;
  accuracy: number;
  train_size: number;
  test_size: number;
  test_positive_count: number;
  test_negative_count: number;
}

export interface TrainingTask {
  id: string;
  status: TaskStatus;
  progress: number;
  selectedFeatures: FeatureId[];
  datasetName: string;
  datasetSize: number;
  logs: TrainingLog[];
  createdAt: string;
  completedAt?: string;
  metrics?: TrainingMetrics;
}

export interface InferenceResult {
  matchProbability: number;
  featuresUsed: FeatureId[];
  featureValues: Record<FeatureId, number>;
  timeMs: number;
}
