/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Network, GraduationCap, Search, BookOpen } from 'lucide-react';

interface HeaderProps {
  activeTab: 'training' | 'testing' | 'documentation';
  setActiveTab: (tab: 'training' | 'testing' | 'documentation') => void;
}

export default function Header({ activeTab, setActiveTab }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-100 bg-white/70 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        
        {/* Logo and Brand */}
        <div 
          className="flex items-center gap-3 cursor-pointer select-none"
          onClick={() => setActiveTab('training')}
          id="brand-logo"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 text-white shadow-md shadow-blue-500/20">
            <Network className="h-5 w-5 animate-pulse" />
          </div>
          <div>
            <h1 className="font-sans text-base font-semibold tracking-tight text-slate-850">
              ML Matcher
            </h1>
            <p className="font-sans text-xs text-slate-400">
              матчинг строк с ML
            </p>
          </div>
        </div>

        {/* Dynamic Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-slate-100/70 p-1 rounded-2xl border border-slate-200/40">
          <button
            id="tab-training"
            onClick={() => setActiveTab('training')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl transition-all duration-300 ${
              activeTab === 'training'
                ? 'bg-white text-blue-600 shadow-sm shadow-blue-500/5 ring-1 ring-slate-100/5'
                : 'text-slate-500 hover:text-slate-900 hover:bg-white/40'
            }`}
          >
            <GraduationCap className="h-4 w-4" />
            <span>Обучение модели</span>
          </button>

          <button
            id="tab-testing"
            onClick={() => setActiveTab('testing')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl transition-all duration-300 ${
              activeTab === 'testing'
                ? 'bg-white text-blue-600 shadow-sm shadow-blue-500/5 ring-1 ring-slate-100/5'
                : 'text-slate-500 hover:text-slate-900 hover:bg-white/40'
            }`}
          >
            <Search className="h-4 w-4" />
            <span>Тестирование</span>
          </button>
        </nav>

        {/* Documentation Tab */}
        <div>
          <button
            id="tab-documentation"
            onClick={() => setActiveTab('documentation')}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl border transition-all duration-300 ${
              activeTab === 'documentation'
                ? 'bg-slate-900 text-white border-slate-950 shadow-md shadow-slate-950/10'
                : 'bg-white text-slate-600 border-slate-200 hover:text-slate-950 hover:bg-slate-50 hover:border-slate-300'
            }`}
          >
            <BookOpen className="h-4 w-4" />
            <span>Документация</span>
          </button>
        </div>

      </div>
    </header>
  );
}
