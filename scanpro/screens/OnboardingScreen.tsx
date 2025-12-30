import React from 'react';

interface OnboardingScreenProps {
  onComplete: () => void;
}

export const OnboardingScreen: React.FC<OnboardingScreenProps> = ({ onComplete }) => {
  return (
    <div className="bg-background-light dark:bg-background-dark font-display h-screen w-full overflow-hidden flex flex-col relative selection:bg-primary/30">
      <div className="absolute top-[-20%] left-[-20%] w-[600px] h-[600px] bg-primary/20 rounded-full blur-[120px] pointer-events-none opacity-50"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[400px] h-[400px] bg-blue-600/10 rounded-full blur-[100px] pointer-events-none opacity-30"></div>

      <div className="relative flex flex-col h-full w-full z-10 max-w-md mx-auto shadow-2xl overflow-hidden">
        <div className="flex-1 relative w-full flex items-center justify-center pt-12 pb-24">
          <div className="relative w-full h-full max-h-[60vh] flex flex-col items-center justify-center p-4">
            <div 
              className="w-full h-full bg-center bg-contain bg-no-repeat transform transition-transform duration-700 hover:scale-105" 
              style={{
                backgroundImage: "url('https://lh3.googleusercontent.com/aida-public/AB6AXuAKAIhi74ErqW-vAMCbZT5v3gk1obpkHepJjLCBs_C5ITnBur82Ubbj5WUmQys0YRa5OVh2VdPEWWk5g02zDZU53DTy1XdKFiEwN6jzWusogKzC-6WiS7tky3DuoD2ddZKMNPB3HTy4O5P53jlXaIaBGQQZ9t33onNYvJjNUqkeYwwhuEtCEUF7uiytYwe4VOgh386NZCJyi-jazaFSpWg-Uq1NtkNV4DfunWvfaSzK_Rf0htDm8YEaAA7gNA1c7SbDc-b8U7jkQh-H')",
                maskImage: "linear-gradient(to bottom, black 80%, transparent 100%)",
                WebkitMaskImage: "linear-gradient(to bottom, black 80%, transparent 100%)"
              }}
            ></div>
            
            <div className="absolute top-1/4 right-10 w-12 h-12 rounded-xl bg-white/5 backdrop-blur-md border border-white/10 shadow-lg rotate-12 flex items-center justify-center animate-pulse">
              <span className="material-symbols-outlined text-primary text-2xl">crop_free</span>
            </div>
            <div className="absolute bottom-1/3 left-8 w-10 h-10 rounded-lg bg-white/5 backdrop-blur-md border border-white/10 shadow-lg -rotate-6 flex items-center justify-center">
              <span className="material-symbols-outlined text-white/70 text-xl">auto_fix_high</span>
            </div>
          </div>
        </div>

        <div className="glass-panel w-full rounded-t-[40px] px-8 pt-10 pb-8 flex flex-col shadow-[0_-10px_60px_-15px_rgba(0,0,0,0.6)] absolute bottom-0 bg-[#101c22]/60">
          <div className="mb-10 text-center">
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-4 leading-tight text-transparent bg-clip-text bg-gradient-to-r from-white to-[#42b6f0]">
              Scan in Seconds
            </h1>
            <p className="text-gray-400 text-base md:text-lg font-normal leading-relaxed px-2">
              Capture crystal clear documents with auto-edge detection and intelligent processing.
            </p>
          </div>

          <div className="flex flex-col gap-8 w-full">
            <div className="flex w-full flex-row items-center justify-center gap-2">
              <div className="h-2 w-8 rounded-full bg-primary shadow-[0_0_10px_rgba(66,182,240,0.5)] transition-all duration-300"></div>
              <div className="h-2 w-2 rounded-full bg-white/20 transition-all duration-300"></div>
              <div className="h-2 w-2 rounded-full bg-white/20 transition-all duration-300"></div>
            </div>

            <div className="flex items-center justify-between mt-2">
              <button onClick={onComplete} className="px-6 py-3 text-gray-400 text-sm font-medium hover:text-white transition-colors duration-200">
                Skip
              </button>
              <button onClick={onComplete} className="group relative flex items-center justify-center overflow-hidden rounded-full bg-gradient-to-r from-primary to-[#2a93c9] text-[#111c22] h-14 pr-2 pl-6 gap-3 shadow-[0_4px_20px_rgba(66,182,240,0.4)] transition-all duration-300 hover:shadow-[0_6px_25px_rgba(66,182,240,0.6)] hover:scale-105 active:scale-95">
                <span className="text-base font-bold tracking-wide text-white">Next</span>
                <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center backdrop-blur-sm group-hover:bg-white/30 transition-colors">
                  <span className="material-symbols-outlined text-white text-[20px]">arrow_forward</span>
                </div>
              </button>
            </div>
          </div>
          <div className="h-6 w-full"></div>
        </div>
      </div>
    </div>
  );
};
