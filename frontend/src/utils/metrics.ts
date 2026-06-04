/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { FeatureId } from '../types';

export function getLevenshteinDistance(s1: string, s2: string): number {
  const len1 = s1.length;
  const len2 = s2.length;
  const dp = Array.from({ length: len1 + 1 }, () => Array(len2 + 1).fill(0));

  for (let i = 0; i <= len1; i++) dp[i][0] = i;
  for (let j = 0; j <= len2; j++) dp[0][j] = j;

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1, // deletion
        dp[i][j - 1] + 1, // insertion
        dp[i - 1][j - 1] + cost // substitution
      );
    }
  }
  return dp[len1][len2];
}

export function getLevenshteinSimilarity(s1: string, s2: string): number {
  const maxLen = Math.max(s1.length, s2.length);
  if (maxLen === 0) return 1.0;
  return 1.0 - getLevenshteinDistance(s1, s2) / maxLen;
}

export function getLevenshteinTokenSortSimilarity(s1: string, s2: string): number {
  // Сортируем слова перед расчётом — порядок слов перестаёт влиять.
  const sortTokens = (str: string) =>
    str
      .toLowerCase()
      .replace(/[^\w\sа-яА-ЯёЁ]/g, ' ')
      .split(/\s+/)
      .filter(Boolean)
      .sort()
      .join(' ');
  return getLevenshteinSimilarity(sortTokens(s1), sortTokens(s2));
}

export function getJaroWinklerSimilarity(s1: string, s2: string): number {
  const len1 = s1.length;
  const len2 = s2.length;
  if (len1 === 0 && len2 === 0) return 1.0;
  if (len1 === 0 || len2 === 0) return 0.0;

  const matchWindow = Math.floor(Math.max(len1, len2) / 2) - 1;
  const s1Matches = Array(len1).fill(false);
  const s2Matches = Array(len2).fill(false);
  
  let matches = 0;
  for (let i = 0; i < len1; i++) {
    const start = Math.max(0, i - matchWindow);
    const end = Math.min(len2 - 1, i + matchWindow);
    for (let j = start; j <= end; j++) {
      if (!s2Matches[j] && s1[i] === s2[j]) {
        s1Matches[i] = true;
        s2Matches[j] = true;
        matches++;
        break;
      }
    }
  }
  
  if (matches === 0) return 0.0;

  let transpositions = 0;
  let k = 0;
  for (let i = 0; i < len1; i++) {
    if (s1Matches[i]) {
      while (!s2Matches[k]) k++;
      if (s1[i] !== s2[k]) transpositions++;
      k++;
    }
  }

  const jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3;
  
  // Winkler scaling
  let prefix = 0;
  const maxPrefix = Math.min(4, Math.min(len1, len2));
  for (let i = 0; i < maxPrefix; i++) {
    if (s1[i] === s2[i]) prefix++;
    else break;
  }
  
  return jaro + prefix * 0.1 * (1.0 - jaro);
}

export function getDiceSimilarity(s1: string, s2: string): number {
  if (s1 === s2) return 1.0;
  if (s1.length < 2 || s2.length < 2) return 0.0;

  const getBigrams = (str: string) => {
    const bigrams = new Set<string>();
    for (let i = 0; i < str.length - 1; i++) {
      bigrams.add(str.substring(i, i + 2).toLowerCase());
    }
    return bigrams;
  };

  const b1 = getBigrams(s1);
  const b2 = getBigrams(s2);
  
  let intersections = 0;
  b1.forEach(bg => {
    if (b2.has(bg)) intersections++;
  });

  return (2.0 * intersections) / (b1.size + b2.size);
}

export function getNumberMatch(s1: string, s2: string): number {
  const d1 = s1.replace(/\D/g, '');
  const d2 = s2.replace(/\D/g, '');

  // If neither has numbers, they match in the "absence of numbers" sense
  if (d1 === '' && d2 === '') return 1.0;
  if (d1 === '' || d2 === '') return 0.0;
  return d1 === d2 ? 1.0 : 0.0;
}

// Парсинг чисел с дробной частью (4.5, 0,5) целиком.
function parseNumbers(text: string): number[] {
  const matches = text.match(/\d+(?:[.,]\d+)?/g) || [];
  return matches.map(n => parseFloat(n.replace(',', '.')));
}

export function getNumberSimilarity(s1: string, s2: string): number {
  // Градуированная близость чисел: учитывает величину различия (4.5≈4 высоко, 500 vs 250 низко).
  const n1 = parseNumbers(s1);
  const n2 = parseNumbers(s2);
  if (n1.length === 0 && n2.length === 0) return 1.0;
  if (n1.length === 0 || n2.length === 0) return 0.0;

  const remaining = [...n2];
  const scores: number[] = [];
  for (const a of n1) {
    if (remaining.length === 0) {
      scores.push(0.0);
      continue;
    }
    let bestIdx = 0;
    let bestRatio = -1.0;
    remaining.forEach((b, idx) => {
      const hi = Math.max(Math.abs(a), Math.abs(b));
      const ratio = hi === 0 ? 1.0 : Math.min(Math.abs(a), Math.abs(b)) / hi;
      if (ratio > bestRatio) {
        bestRatio = ratio;
        bestIdx = idx;
      }
    });
    scores.push(bestRatio);
    remaining.splice(bestIdx, 1);
  }
  remaining.forEach(() => scores.push(0.0));
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}

export function getWordIntersection(s1: string, s2: string): number {
  const tokenize = (str: string) => {
    return new Set(
      str
        .toLowerCase()
        .replace(/[^\w\sа-яА-ЯёЁ]/g, ' ')
        .split(/\s+/)
        .filter(Boolean)
    );
  };

  const w1 = tokenize(s1);
  const w2 = tokenize(s2);
  
  if (w1.size === 0 && w2.size === 0) return 1.0;
  if (w1.size === 0 || w2.size === 0) return 0.0;

  let intersect = 0;
  w1.forEach(w => {
    if (w2.has(w)) intersect++;
  });

  // Sorensen-Dice similarity on word tokens
  return (2.0 * intersect) / (w1.size + w2.size);
}

export function getLengthDiff(s1: string, s2: string): number {
  return Math.abs(s1.length - s2.length);
}

export function getLengthDiffNormalized(s1: string, s2: string): number {
  const maxLen = Math.max(s1.length, s2.length);
  if (maxLen === 0) return 1.0;
  return 1.0 - Math.abs(s1.length - s2.length) / maxLen;
}

export function getTfIdfCosineSimilarity(s1: string, s2: string): number {
  if (s1 === s2) return 1.0;
  
  const getTrigrams = (str: string) => {
    const trigrams: Record<string, number> = {};
    for (let i = 0; i < str.length - 2; i++) {
      const tri = str.substring(i, i + 3).toLowerCase();
      trigrams[tri] = (trigrams[tri] || 0) + 1;
    }
    return trigrams;
  };

  const t1 = getTrigrams(s1);
  const t2 = getTrigrams(s2);

  let dotProduct = 0;
  let mag1 = 0;
  let mag2 = 0;

  for (const tri of Object.keys(t1)) {
    mag1 += t1[tri] * t1[tri];
  }
  for (const tri of Object.keys(t2)) {
    mag2 += t2[tri] * t2[tri];
  }

  for (const tri of Object.keys(t1)) {
    if (t2[tri]) {
      dotProduct += t1[tri] * t2[tri];
    }
  }

  if (mag1 === 0 || mag2 === 0) return 0.0;
  return dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2));
}

export function calculateFeatureValue(featureId: FeatureId, s1: string, s2: string): number {
  switch (featureId) {
    case 'levenshtein':
      return getLevenshteinSimilarity(s1, s2);
    case 'levenshtein_token_sort':
      return getLevenshteinTokenSortSimilarity(s1, s2);
    case 'jaro_winkler':
      return getJaroWinklerSimilarity(s1, s2);
    case 'dice':
      return getDiceSimilarity(s1, s2);
    case 'number_match':
      return getNumberMatch(s1, s2);
    case 'number_similarity':
      return getNumberSimilarity(s1, s2);
    case 'word_intersection':
      return getWordIntersection(s1, s2);
    case 'length_diff':
      // Give absolute length diff as a feature, but for prediction similarity, normalize it
      return getLengthDiff(s1, s2);
    case 'tfidf_cosine':
      return getTfIdfCosineSimilarity(s1, s2);
    case 'embedding_cosine':
      return 0.0; // Семантические эмбеддинги доступны только на сервере
    default:
      return 0.0;
  }
}

/**
 * Computes match probability based on selected features.
 * We can simulate a CatBoost model using a weighted logistic function
 * that depends on the calculated similarity scores.
 */
export function computeMatchProbability(
  s1: string,
  s2: string,
  features: FeatureId[]
): {
  probability: number;
  featureValues: Record<FeatureId, number>;
} {
  const values: Record<FeatureId, number> = {} as any;
  let scoreSum = 0;
  let weightSum = 0;

  // Define weights for each metric
  const weights: Record<FeatureId, number> = {
    levenshtein: 1.5,
    levenshtein_token_sort: 1.5,
    jaro_winkler: 1.8,
    dice: 1.2,
    number_match: 2.2, // If numbers mismatch, this is highly indicative of mismatch
    number_similarity: 2.2, // Градуированная близость чисел — высокий вес
    word_intersection: 1.4,
    length_diff: -0.1, // Absolute length difference weighs negatively
    tfidf_cosine: 2.0,
    embedding_cosine: 0, // Эмбеддинги считаются только на сервере — в превью не участвуют
  };

  for (const fid of features) {
    const rawVal = calculateFeatureValue(fid, s1, s2);
    values[fid] = rawVal;
    
    if (fid === 'length_diff') {
      // Use standard normalized representation for weighing, where length difference reduces probability
      const normVal = getLengthDiffNormalized(s1, s2); // 1 is perfect match, 0 is very long difference
      scoreSum += normVal * 1.5; // virtual weight
      weightSum += 1.5;
    } else if (fid === 'number_match') {
      // If there are digits and they don't match, heavily pull similarity down. Keep it as 0/1 indicator
      if (rawVal === 0.0) {
        scoreSum += 0.0;
        weightSum += weights[fid];
      } else {
        scoreSum += 1.0 * weights[fid];
        weightSum += weights[fid];
      }
    } else {
      scoreSum += rawVal * weights[fid];
      weightSum += weights[fid];
    }
  }

  let probability = weightSum > 0 ? scoreSum / weightSum : 0.0;

  // Let's perform a non-linear adjustment (sigmoid-like) to pull highly similar things to 100%,
  // and dissimilar things to 0%, mimicking decision tree ensembles.
  if (probability > 0.85) {
    probability = 0.85 + (probability - 0.85) * 1.0; // Scaled high
  } else if (probability < 0.3) {
    probability = probability * 0.5; // Depressed low
  }
  
  // Cap between 0 and 1
  probability = Math.max(0, Math.min(1, probability));

  return {
    probability,
    featureValues: values,
  };
}
