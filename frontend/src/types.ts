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
  | 'length_diff'
  | 'tfidf_cosine'
  | 'embedding_cosine';

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
    description: 'Учитывает величину различия дозировок: 4.5≈4 — высокий балл, 500 vs 250 — низкий. Чтобы цифры весили больше.',
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
];

export type TaskStatus = 'idle' | 'preparing' | 'tfidf' | 'negatives' | 'training' | 'completed' | 'failed';

export interface TrainingLog {
  timestamp: string;
  type: 'info' | 'warn' | 'success' | 'iteration';
  message: string;
  progress: number;
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
}

export interface InferenceResult {
  matchProbability: number;
  featuresUsed: FeatureId[];
  featureValues: Record<FeatureId, number>;
  timeMs: number;
}
