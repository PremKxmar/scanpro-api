import React from 'react';
import { ThemeContextType } from '../types';

interface SettingsScreenProps {
    onBack: () => void;
    themeContext: ThemeContextType;
}

export const SettingsScreen: React.FC<SettingsScreenProps> = ({ onBack, themeContext }) => {
    return (
        <div className="bg-background-dark text-white font-display antialiased overflow-x-hidden selection:bg-primary selection:text-white min-h-screen flex flex-col pb-32">
            {/* Ambient Background */}
            <div className="fixed top-0 right-0 w-[300px] h-[300px] bg-primary/10 rounded-full blur-[120px] -translate-y-1/3 translate-x-1/3 pointer-events-none"></div>

            <header className="sticky top-0 z-50 px-6 py-6 flex items-center gap-4 bg-background-dark/90 backdrop-blur-xl border-b border-white/5">
                <button onClick={onBack} className="flex size-10 items-center justify-center rounded-full hover:bg-white/10 transition-colors">
                    <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>arrow_back</span>
                </button>
                <h1 className="text-white text-2xl font-bold tracking-tight">Settings</h1>
            </header>

            <main className="flex-1 px-6 py-2 space-y-8 z-10 relative">

                {/* Account Section */}
                <section>
                    <div className="flex items-center gap-4 p-4 rounded-[1.5rem] bg-surface-dark border border-white/5 mb-6">
                        <div className="relative">
                            <img className="size-16 rounded-full object-cover border-2 border-primary/30" src="https://lh3.googleusercontent.com/aida-public/AB6AXuAYIYrOcFX_UQsaWUdbEKzjP-lS4kWp_uubGdMKHMMTtBj8jz56l07OlykT1_bPT7js5WfQyO4OQ4Xm5lAL4oLCKJg9LoPQ18EYi39JYT3nDPSiL8Hwm53-OM4GFitY27pRAqdgeLFILkY0nPu76gvZhk4n0_jm1qUCOOGnli_Uf2t-p_ZToNF6rXs2n7YLF0afzkPMLh5fxva-kzjC1IsA8Mzp9SZhXMAnoj25mEiPSnD701BUFA8OIiSMdxqmJ2du4tsb59KjOd_V" alt="Profile" />
                            <div className="absolute bottom-0 right-0 bg-primary border-2 border-background-dark rounded-full p-1">
                                <span className="material-symbols-outlined text-white text-[12px] block">edit</span>
                            </div>
                        </div>
                        <div className="flex-1">
                            <h2 className="text-lg font-bold text-white">Alex Morgan</h2>
                            <p className="text-sm text-gray-400">alex.morgan@example.com</p>
                            <div className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-yellow-500/10 border border-yellow-500/20">
                                <span className="material-symbols-outlined text-yellow-500 text-[12px] filled">star</span>
                                <span className="text-[10px] font-bold text-yellow-500 uppercase tracking-wide">Premium</span>
                            </div>
                        </div>
                    </div>
                </section>

                {/* General Settings */}
                <section>
                    <h3 className="text-gray-500 text-xs font-bold uppercase tracking-widest px-2 mb-3">General</h3>
                    <div className="bg-surface-dark border border-white/5 rounded-[1.5rem] overflow-hidden">
                        <div className="flex items-center justify-between p-4 border-b border-white/5 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
                                    <span className="material-symbols-outlined">center_focus_strong</span>
                                </div>
                                <span className="text-[15px] font-medium">Scan Quality</span>
                            </div>
                            <span className="text-sm text-gray-400 flex items-center gap-1">
                                High
                                <span className="material-symbols-outlined text-[16px]">chevron_right</span>
                            </span>
                        </div>

                        <div className="flex items-center justify-between p-4 border-b border-white/5 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400">
                                    <span className="material-symbols-outlined">auto_fix_high</span>
                                </div>
                                <span className="text-[15px] font-medium">Auto-enhance</span>
                            </div>
                            <div className="w-11 h-6 bg-primary rounded-full relative">
                                <div className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full shadow-sm"></div>
                            </div>
                        </div>

                        <div className="flex items-center justify-between p-4 border-b border-white/5 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-green-500/10 text-green-400">
                                    <span className="material-symbols-outlined">save</span>
                                </div>
                                <span className="text-[15px] font-medium">Save to Gallery</span>
                            </div>
                            <div className="w-11 h-6 bg-white/10 rounded-full relative">
                                <div className="absolute left-1 top-1 w-4 h-4 bg-white/50 rounded-full shadow-sm"></div>
                            </div>
                        </div>

                        <div className="flex items-center justify-between p-4 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-pink-500/10 text-pink-400">
                                    <span className="material-symbols-outlined">dark_mode</span>
                                </div>
                                <span className="text-[15px] font-medium">Appearance</span>
                            </div>
                            <span className="text-sm text-gray-400 flex items-center gap-1">
                                Dark
                                <span className="material-symbols-outlined text-[16px]">chevron_right</span>
                            </span>
                        </div>
                    </div>
                </section>

                {/* Data & Storage */}
                <section>
                    <h3 className="text-gray-500 text-xs font-bold uppercase tracking-widest px-2 mb-3">Data</h3>
                    <div className="bg-surface-dark border border-white/5 rounded-[1.5rem] overflow-hidden">
                        <div className="flex items-center justify-between p-4 border-b border-white/5 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
                                    <span className="material-symbols-outlined">cloud_sync</span>
                                </div>
                                <span className="text-[15px] font-medium">Cloud Sync</span>
                            </div>
                            <span className="text-sm text-gray-400 flex items-center gap-1">
                                On
                                <span className="material-symbols-outlined text-[16px]">chevron_right</span>
                            </span>
                        </div>

                        <div className="flex items-center justify-between p-4 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-orange-500/10 text-orange-400">
                                    <span className="material-symbols-outlined">folder_delete</span>
                                </div>
                                <span className="text-[15px] font-medium">Clear Cache</span>
                            </div>
                            <span className="text-sm text-gray-400">128 MB</span>
                        </div>
                    </div>
                </section>

                {/* Support */}
                <section>
                    <h3 className="text-gray-500 text-xs font-bold uppercase tracking-widest px-2 mb-3">Support</h3>
                    <div className="bg-surface-dark border border-white/5 rounded-[1.5rem] overflow-hidden">
                        <div className="flex items-center justify-between p-4 border-b border-white/5 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                                    <span className="material-symbols-outlined">help</span>
                                </div>
                                <span className="text-[15px] font-medium">Help Center</span>
                            </div>
                            <span className="material-symbols-outlined text-gray-500 text-[20px]">open_in_new</span>
                        </div>

                        <div className="flex items-center justify-between p-4 border-b border-white/5 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-red-500/10 text-red-400">
                                    <span className="material-symbols-outlined">lock</span>
                                </div>
                                <span className="text-[15px] font-medium">Privacy Policy</span>
                            </div>
                        </div>

                        <div className="flex items-center justify-between p-4 active:bg-white/5 transition-colors cursor-pointer">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-gray-500/10 text-gray-400">
                                    <span className="material-symbols-outlined">info</span>
                                </div>
                                <span className="text-[15px] font-medium">About</span>
                            </div>
                            <span className="text-sm text-gray-400">v4.0.1</span>
                        </div>
                    </div>
                </section>

                <button className="w-full py-4 rounded-[1.5rem] border border-red-500/20 text-red-500 text-[15px] font-bold bg-red-500/5 active:bg-red-500/10 transition-colors">
                    Log Out
                </button>

            </main>
        </div>
    );
};
