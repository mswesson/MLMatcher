/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import Header from './components/Header';
import StepTraining from './components/StepTraining';
import StepInference from './components/StepInference';
import StepBatchEval from './components/StepBatchEval';
import Documentation from './components/Documentation';

export default function App() {
  const [activeTab, setActiveTab] = useState<'training' | 'testing' | 'batch' | 'documentation'>('training');

  return (
    <div className="relative min-h-screen bg-slate-50 overflow-hidden font-sans antialiased text-slate-800">
      
      {/* 1. ARCHITECTURAL BG DECORATION SHAPES */}
      <div className="absolute top-0 left-0 w-full h-[600px] bg-gradient-to-b from-blue-50/75 via-transparent to-transparent pointer-events-none -z-10" />
      
      {/* Dynamic abstract radial meshes matching the drawing backgrounds */}
      <div className="absolute top-[-250px] right-[-100px] w-[500px] h-[500px] rounded-full bg-blue-400/10 blur-[100px] pointer-events-none -z-10 animate-pulse" style={{ animationDuration: '10s' }} />
      <div className="absolute bottom-[10%] left-[-200px] w-[600px] h-[600px] rounded-full bg-indigo-300/10 blur-[120px] pointer-events-none -z-10 animate-pulse" style={{ animationDuration: '15s' }} />

      {/* 2. MAIN HEADER */}
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* 3. CORE ROUTER PAGE ENCLOSURE */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        
        <AnimatePresence mode="wait">
          {activeTab === 'training' && (
            <motion.div
              key="training-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.35, ease: 'easeInOut' }}
            >
              <StepTraining />
            </motion.div>
          )}

          {activeTab === 'testing' && (
            <motion.div
              key="testing-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.35, ease: 'easeInOut' }}
            >
              <StepInference />
            </motion.div>
          )}

          {activeTab === 'batch' && (
            <motion.div
              key="batch-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.35, ease: 'easeInOut' }}
            >
              <StepBatchEval />
            </motion.div>
          )}

          {activeTab === 'documentation' && (
            <motion.div
              key="docs-tab"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.35, ease: 'easeInOut' }}
            >
              <Documentation />
            </motion.div>
          )}
        </AnimatePresence>

      </main>

      {/* 4. FOOTER CREDITS */}
      <footer className="w-full border-t border-slate-100 bg-white/40 py-6 mt-16 text-center text-xs text-slate-400">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="font-sans">
            &copy; {new Date().getFullYear()} ML Matcher. Все права защищены.
          </p>
          <div className="flex items-center gap-4">
            <span className="font-sans">ML Core v1.0.0 (CatBoost)</span>
            <span className="h-4 w-px bg-slate-200" />
            <span className="font-mono text-[10px]">FastAPI & React Slice Architectural Framework</span>
          </div>
        </div>
      </footer>

    </div>
  );
}
