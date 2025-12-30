import React from 'react';
import api from '../services/api';

interface ViewerScreenProps {
   onBack: () => void;
   title: string;
   size: string;
   thumbnail?: string;
}

export const ViewerScreen: React.FC<ViewerScreenProps> = ({ onBack, title, size, thumbnail }) => {

   const handleShare = async () => {
      if (!thumbnail) return;
      try {
         const blob = api.base64ToBlob(thumbnail);
         const file = new File([blob], title, { type: 'image/jpeg' });
         if (navigator.share) {
            await navigator.share({ files: [file], title });
         }
      } catch (e) {
         console.log('Share failed');
      }
   };

   const handleDownload = () => {
      if (thumbnail) {
         api.downloadBase64(thumbnail, title);
      }
   };

   return (
      <div className="bg-background-dark font-display text-white overflow-hidden h-screen w-full relative flex flex-col">

         {/* Top Bar */}
         <header className="fixed top-0 left-0 right-0 z-40 px-4 pt-12 pb-4 flex items-center justify-between bg-gradient-to-b from-background-dark/90 to-transparent">
            <button onClick={onBack} className="flex size-10 items-center justify-center rounded-full hover:bg-white/10 active:bg-white/20 transition-colors">
               <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>arrow_back_ios_new</span>
            </button>
            <div className="flex flex-col items-center">
               <h1 className="text-white text-[15px] font-bold tracking-wide">{title}</h1>
               <span className="text-gray-400 text-[11px] font-medium">Local • {size}</span>
            </div>
            <button className="flex size-10 items-center justify-center rounded-full hover:bg-white/10 active:bg-white/20 transition-colors">
               <span className="material-symbols-outlined text-white" style={{ fontSize: 24 }}>more_horiz</span>
            </button>
         </header>

         {/* Main Content */}
         <main className="relative z-10 flex-1 w-full flex flex-col items-center justify-center pb-24 pt-20 px-6">
            <div className="relative w-full max-w-sm aspect-[3/4] group">
               {/* Document Card */}
               <div className="relative w-full h-full bg-white rounded-lg shadow-2xl overflow-hidden">
                  {thumbnail ? (
                     <img src={thumbnail} alt={title} className="w-full h-full object-contain" />
                  ) : (
                     <div className="w-full h-full flex items-center justify-center bg-gray-800">
                        <span className="material-symbols-outlined text-gray-500 text-6xl">description</span>
                     </div>
                  )}
               </div>
            </div>

            {/* Pagination Dots */}
            <div className="mt-8 flex flex-row items-center justify-center gap-2.5">
               <div className="size-2 rounded-full bg-primary"></div>
               <div className="size-2 rounded-full bg-white/20"></div>
               <div className="size-2 rounded-full bg-white/20"></div>
               <div className="size-2 rounded-full bg-white/20"></div>
            </div>
         </main>

         {/* Bottom Floating Bar */}
         <div className="fixed bottom-10 left-0 right-0 z-40 flex justify-center px-6">
            <div className="bg-[#1e1f2e] border border-white/5 rounded-[2rem] px-6 py-2 flex items-center gap-2 shadow-2xl w-full max-w-xs justify-between">
               <button className="flex flex-col items-center gap-1 group p-2 min-w-[56px]">
                  <div className="size-10 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-white/10 transition-colors">
                     <span className="material-symbols-outlined text-white" style={{ fontSize: 20 }}>share</span>
                  </div>
                  <span className="text-[10px] font-medium text-gray-400">Share</span>
               </button>

               <button className="flex flex-col items-center gap-1 group p-2 min-w-[56px]">
                  <div className="size-10 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-white/10 transition-colors">
                     <span className="material-symbols-outlined text-white" style={{ fontSize: 20 }}>edit</span>
                  </div>
                  <span className="text-[10px] font-medium text-gray-400">Edit</span>
               </button>

               <button className="flex flex-col items-center gap-1 group p-2 min-w-[56px]">
                  <div className="size-10 rounded-full bg-white/5 flex items-center justify-center group-hover:bg-white/10 transition-colors">
                     <span className="material-symbols-outlined text-white" style={{ fontSize: 20 }}>delete</span>
                  </div>
                  <span className="text-[10px] font-medium text-gray-400">Delete</span>
               </button>

               <button className="flex flex-col items-center gap-1 group p-2 min-w-[56px]">
                  <div className="size-10 rounded-full bg-primary flex items-center justify-center shadow-lg shadow-primary/30 group-hover:bg-primary-dark transition-colors">
                     <span className="material-symbols-outlined text-white" style={{ fontSize: 20 }}>ios_share</span>
                  </div>
                  <span className="text-[10px] font-medium text-primary">Export</span>
               </button>
            </div>
         </div>
      </div>
   );
};
