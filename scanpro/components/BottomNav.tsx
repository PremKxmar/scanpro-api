import React from 'react';
import { Screen } from '../types';

interface BottomNavProps {
  currentScreen: Screen;
  onNavigate: (screen: Screen) => void;
}

export const BottomNav: React.FC<BottomNavProps> = ({ currentScreen, onNavigate }) => {
  return (
    <div className="absolute bottom-0 left-0 right-0 z-50">
      <div className="mx-5 mb-5 h-20 rounded-[2.5rem] glass-panel flex items-center justify-around px-2 shadow-2xl bg-surface-dark/90 backdrop-blur-xl border border-white/10">
        <button 
          onClick={() => onNavigate('library')}
          className={`flex flex-col items-center justify-center gap-1 w-16 group transition-all duration-300 ${currentScreen === 'library' ? 'scale-110' : 'opacity-60 hover:opacity-100'}`}
        >
          <div className={`${currentScreen === 'library' ? 'bg-primary/20 shadow-[0_0_15px_rgba(92,107,250,0.3)]' : ''} p-2 rounded-full mb-0.5 transition-colors`}>
            <span className={`material-symbols-outlined text-[24px] ${currentScreen === 'library' ? 'text-primary font-variation-settings-\'FILL\'-1' : 'text-white'}`}>
              grid_view
            </span>
          </div>
          <span className={`text-[10px] font-bold ${currentScreen === 'library' ? 'text-white' : 'text-white/60'}`}>Library</span>
        </button>

        <button 
          onClick={() => onNavigate('tools')}
          className={`flex flex-col items-center justify-center gap-1 w-16 group transition-all duration-300 ${currentScreen === 'tools' ? 'scale-110' : 'opacity-60 hover:opacity-100'}`}
        >
          <div className={`${currentScreen === 'tools' ? 'bg-primary/20 shadow-[0_0_15px_rgba(92,107,250,0.3)]' : ''} p-2 rounded-full mb-0.5 transition-colors`}>
            <span className={`material-symbols-outlined text-[24px] ${currentScreen === 'tools' ? 'text-primary font-variation-settings-\'FILL\'-1' : 'text-white'}`}>
              build
            </span>
          </div>
          <span className={`text-[10px] font-bold ${currentScreen === 'tools' ? 'text-white' : 'text-white/60'}`}>Tools</span>
        </button>

        <button 
          onClick={() => onNavigate('settings')}
          className={`flex flex-col items-center justify-center gap-1 w-16 group transition-all duration-300 ${currentScreen === 'settings' ? 'scale-110' : 'opacity-60 hover:opacity-100'}`}
        >
          <div className={`${currentScreen === 'settings' ? 'bg-primary/20 shadow-[0_0_15px_rgba(92,107,250,0.3)]' : ''} p-2 rounded-full mb-0.5 transition-colors`}>
            <span className={`material-symbols-outlined text-[24px] ${currentScreen === 'settings' ? 'text-primary font-variation-settings-\'FILL\'-1' : 'text-white'}`}>
              settings
            </span>
          </div>
          <span className={`text-[10px] font-bold ${currentScreen === 'settings' ? 'text-white' : 'text-white/60'}`}>Settings</span>
        </button>
      </div>
    </div>
  );
};
