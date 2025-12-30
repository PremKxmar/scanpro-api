import React from 'react';

export const ToolsScreen: React.FC = () => {
  const tools = [
    { icon: 'text_fields', title: 'Text Recognition', desc: 'Convert image to text', color: 'text-blue-400', bg: 'bg-blue-400/10' },
    { icon: 'ink_pen', title: 'Sign Document', desc: 'Add e-signature', color: 'text-purple-400', bg: 'bg-purple-400/10' },
    { icon: 'badge', title: 'ID Card Scan', desc: 'Front & back on one page', color: 'text-green-400', bg: 'bg-green-400/10' },
    { icon: 'qr_code_scanner', title: 'QR Scanner', desc: 'Read codes instantly', color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
    { icon: 'picture_as_pdf', title: 'Import PDF', desc: 'Edit existing files', color: 'text-red-400', bg: 'bg-red-400/10' },
    { icon: 'image', title: 'Import Images', desc: 'Create PDF from photos', color: 'text-orange-400', bg: 'bg-orange-400/10' },
    { icon: 'join_inner', title: 'Merge PDF', desc: 'Combine multiple files', color: 'text-cyan-400', bg: 'bg-cyan-400/10' },
    { icon: 'lock', title: 'Protect PDF', desc: 'Add password security', color: 'text-pink-400', bg: 'bg-pink-400/10' },
  ];

  return (
    <div className="bg-background-dark h-full min-h-screen w-full relative overflow-y-auto no-scrollbar pb-32">
       {/* Background Effects */}
       <div className="fixed top-0 left-0 w-full h-96 bg-primary/5 rounded-full blur-[100px] -translate-y-1/2 pointer-events-none"></div>
       
       <header className="sticky top-0 z-40 px-6 pt-12 pb-4 bg-background-dark/80 backdrop-blur-xl border-b border-white/5">
        <h1 className="text-white text-2xl font-bold tracking-tight">Tools</h1>
        <p className="text-gray-400 text-sm mt-1">Advanced PDF & Image Utilities</p>
       </header>

       <div className="p-6 grid grid-cols-2 gap-4">
          {tools.map((tool, index) => (
            <button key={index} className="flex flex-col items-start p-4 rounded-[1.5rem] bg-surface-dark border border-white/5 hover:bg-surface-highlight hover:border-white/10 transition-all active:scale-[0.98] group relative overflow-hidden">
                <div className={`p-3 rounded-2xl ${tool.bg} mb-3 group-hover:scale-110 transition-transform duration-300`}>
                    <span className={`material-symbols-outlined ${tool.color} text-[28px]`}>{tool.icon}</span>
                </div>
                <h3 className="text-white text-[15px] font-semibold mb-0.5">{tool.title}</h3>
                <p className="text-gray-500 text-[11px] font-medium leading-tight text-left">{tool.desc}</p>
                
                {/* Hover Glow */}
                <div className="absolute -right-4 -bottom-4 w-16 h-16 bg-white/5 rounded-full blur-2xl group-hover:bg-white/10 transition-colors"></div>
            </button>
          ))}
       </div>

       {/* Promo Card */}
       <div className="mx-6 p-5 rounded-[2rem] bg-gradient-to-br from-primary/20 to-purple-500/20 border border-white/10 relative overflow-hidden">
          <div className="relative z-10">
            <div className="flex items-center gap-2 mb-2">
                <span className="material-symbols-outlined text-primary text-[20px]">diamond</span>
                <span className="text-primary text-xs font-bold tracking-wide uppercase">Pro Feature</span>
            </div>
            <h3 className="text-white text-lg font-bold mb-1">Cloud Sync</h3>
            <p className="text-gray-300 text-xs mb-4 max-w-[200px]">Backup your documents and access them from any device.</p>
            <button className="px-5 py-2 bg-primary text-white text-xs font-bold rounded-full shadow-lg shadow-primary/30 active:scale-95 transition-transform">
                Upgrade Now
            </button>
          </div>
          <div className="absolute right-[-20px] bottom-[-20px] opacity-20">
              <span className="material-symbols-outlined text-[120px] text-white">cloud_upload</span>
          </div>
       </div>
    </div>
  );
};
