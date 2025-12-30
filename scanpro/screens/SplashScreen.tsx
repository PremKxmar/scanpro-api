import React, { useEffect } from 'react';

interface SplashScreenProps {
  onFinish: () => void;
}

export const SplashScreen: React.FC<SplashScreenProps> = ({ onFinish }) => {
  useEffect(() => {
    const timer = setTimeout(onFinish, 2500);
    return () => clearTimeout(timer);
  }, [onFinish]);

  return (
    <div className="relative flex h-screen w-full flex-col items-center justify-center overflow-hidden bg-background-dark">
      {/* Background Gradient Mesh */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[10%] w-[80%] h-[40%] bg-primary/20 blur-[100px] rounded-full mix-blend-screen"></div>
        <div className="absolute -bottom-[10%] -right-[10%] w-[60%] h-[50%] bg-cyan-500/10 blur-[120px] rounded-full mix-blend-screen"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[40rem] h-[40rem] bg-primary/5 blur-[80px] rounded-full"></div>
      </div>

      <div className="relative z-10 flex flex-col items-center justify-center w-full px-8 animate-fadeIn">
        <div className="relative group mb-10">
          <div className="absolute -inset-4 bg-primary/40 rounded-full blur-2xl opacity-60 animate-pulse"></div>
          <div className="relative flex items-center justify-center w-36 h-36 bg-white/5 backdrop-blur-2xl border border-white/10 rounded-full shadow-[0_0_40px_-10px_rgba(100,103,242,0.3)]">
            <div className="absolute inset-2 border border-white/5 rounded-full"></div>
            <div 
              className="w-16 h-16 bg-contain bg-center bg-no-repeat opacity-90 drop-shadow-[0_0_15px_rgba(100,103,242,0.6)]" 
              style={{
                backgroundImage: "url('https://lh3.googleusercontent.com/aida-public/AB6AXuCTK8tiItqKqpplwC3BCYJVhmH-aB2RDZnGtdEyY-TaUjITvNqwQn05lP5j34GzxTT6Svg0ECnIsuAumU9VD16D4AL-4nXgNGw03g6Wuzf1RiM9cVtDXdYgvEwfJg0slEglmc2yHqq5W9aIdHZTJPMOl4n5GSDWxUlXI4bOE5sOl_SMB3tlNTWN1yJdW9eixHnUrbv9blUD-lEYR9RvrI1HajrndZ814xIAmON6CgS0_mCS8_oZHpzz9UtG-nq73svsb0AiA8DeAKZd')",
                filter: "drop-shadow(0px 10px 10px rgba(0,0,0,0.5))"
              }}
            ></div>
          </div>
        </div>

        <div className="flex flex-col items-center text-center space-y-3">
          <h1 className="text-5xl md:text-6xl font-black tracking-tight leading-tight pb-2">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-[#8b8df7] to-cyan-400 drop-shadow-sm">
              ScanPro
            </span>
          </h1>
          <p className="text-[#9293c9] text-xs md:text-sm font-medium tracking-[0.2em] uppercase leading-relaxed opacity-80">
            Scan smarter. Shadow-free.
          </p>
        </div>

        <div className="absolute bottom-16 left-0 right-0 flex justify-center">
          <div className="w-8 h-8 text-primary/60 animate-spin">
            <span className="material-symbols-outlined text-3xl">progress_activity</span>
          </div>
        </div>

        <div className="absolute bottom-6 left-0 right-0 flex justify-center">
          <p className="text-[#9293c9]/30 text-[10px] font-medium tracking-wide">v4.0.1</p>
        </div>
      </div>
    </div>
  );
};
