import React, { useState, useEffect } from 'react';
import { loadDocuments, removeDocument, getDocumentCount } from '../store';
import { DocumentItem } from '../types';

interface LibraryScreenProps {
  onScan: () => void;
  onViewDoc: (id: string) => void;
}

export const LibraryScreen: React.FC<LibraryScreenProps> = ({ onScan, onViewDoc }) => {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [filter, setFilter] = useState<'all' | 'pdf' | 'jpg'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Load documents on mount and when returning to this screen
  useEffect(() => {
    refreshDocuments();
  }, []);

  const refreshDocuments = () => {
    const docs = loadDocuments();
    setDocuments(docs);
  };

  // Filter documents
  const filteredDocs = documents.filter(doc => {
    if (filter !== 'all' && doc.type !== filter) return false;
    if (searchQuery && !doc.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const handleDeleteDoc = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirm('Delete this document?')) {
      removeDocument(id);
      refreshDocuments();
    }
  };

  return (
    <div className="relative mx-auto flex h-full min-h-screen w-full max-w-md flex-col bg-background-light dark:bg-background-dark shadow-2xl overflow-y-auto">
      {/* Background Gradients */}
      <div className="fixed top-[-10%] left-[-10%] h-[500px] w-[500px] rounded-full bg-primary/20 blur-[100px] pointer-events-none"></div>
      <div className="fixed bottom-[-10%] right-[-10%] h-[400px] w-[400px] rounded-full bg-blue-600/10 blur-[100px] pointer-events-none"></div>

      {/* Top Bar */}
      <div className="sticky top-0 z-40 flex flex-col gap-4 pb-2 pt-12 px-5 glass-panel border-b-0 rounded-b-[2rem]">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">Library</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 font-medium">{documents.length} Documents</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={refreshDocuments}
              className="flex size-10 items-center justify-center rounded-full bg-black/5 dark:bg-white/5 text-slate-700 dark:text-white hover:bg-black/10 dark:hover:bg-white/10 transition active:scale-95 border border-black/5 dark:border-white/5"
            >
              <span className="material-symbols-outlined text-[20px]">refresh</span>
            </button>
            <button className="flex size-10 items-center justify-center rounded-full overflow-hidden border border-black/10 dark:border-white/10 bg-primary/20">
              <span className="material-symbols-outlined text-primary text-[20px]">person</span>
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="relative w-full group">
          <div className="absolute inset-y-0 left-0 flex items-center pl-4 pointer-events-none">
            <span className="material-symbols-outlined text-gray-400 group-focus-within:text-primary transition-colors">search</span>
          </div>
          <input
            className="block w-full rounded-2xl border-none bg-white dark:bg-black/20 py-3.5 pl-11 pr-4 text-sm text-slate-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-primary/50 dark:focus:bg-black/30 transition-all shadow-sm dark:shadow-inner"
            placeholder="Search documents..."
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto px-5 pt-4 pb-32 no-scrollbar relative z-10">
        {/* Filter Buttons */}
        <div className="flex gap-3 mb-6 overflow-x-auto no-scrollbar pb-2">
          <button
            onClick={() => setFilter('all')}
            className={`whitespace-nowrap px-5 py-2 rounded-full text-sm font-semibold transition ${filter === 'all'
                ? 'bg-primary text-white shadow-glow'
                : 'bg-white dark:bg-white/5 border border-slate-200 dark:border-white/5 text-slate-600 dark:text-gray-300 hover:bg-slate-50 dark:hover:bg-white/10'
              }`}
          >
            All Scans
          </button>
          <button
            onClick={() => setFilter('pdf')}
            className={`whitespace-nowrap px-5 py-2 rounded-full text-sm font-semibold transition ${filter === 'pdf'
                ? 'bg-primary text-white shadow-glow'
                : 'bg-white dark:bg-white/5 border border-slate-200 dark:border-white/5 text-slate-600 dark:text-gray-300 hover:bg-slate-50 dark:hover:bg-white/10'
              }`}
          >
            PDFs
          </button>
          <button
            onClick={() => setFilter('jpg')}
            className={`whitespace-nowrap px-5 py-2 rounded-full text-sm font-semibold transition ${filter === 'jpg'
                ? 'bg-primary text-white shadow-glow'
                : 'bg-white dark:bg-white/5 border border-slate-200 dark:border-white/5 text-slate-600 dark:text-gray-300 hover:bg-slate-50 dark:hover:bg-white/10'
              }`}
          >
            Images
          </button>
        </div>

        {/* Empty State */}
        {filteredDocs.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 px-4">
            <div className="size-20 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <span className="material-symbols-outlined text-primary text-4xl">document_scanner</span>
            </div>
            <h3 className="text-lg font-bold text-white mb-2">No documents yet</h3>
            <p className="text-gray-400 text-sm text-center mb-6">
              Tap the + button below to scan your first document
            </p>
            <button
              onClick={onScan}
              className="px-6 py-3 bg-primary rounded-full text-white font-bold shadow-lg shadow-primary/30 active:scale-95 transition"
            >
              Start Scanning
            </button>
          </div>
        )}

        {/* Document Grid */}
        {filteredDocs.length > 0 && (
          <div className="grid grid-cols-2 gap-4">
            {filteredDocs.map((doc) => (
              <div
                key={doc.id}
                onClick={() => onViewDoc(doc.id)}
                className="group relative flex flex-col gap-3 p-3 rounded-[2rem] glass-card dark:hover:bg-white/10 hover:bg-white transition-all duration-300 active:scale-[0.98] cursor-pointer"
              >
                <div className="relative aspect-[3/4] w-full overflow-hidden rounded-[1.5rem] bg-gray-200 dark:bg-gray-800 shadow-lg group-hover:shadow-primary/20 transition-all">
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent z-10"></div>

                  {/* Thumbnail */}
                  {doc.thumbnail ? (
                    <img
                      src={doc.thumbnail}
                      alt={doc.title}
                      className="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center bg-gray-700">
                      <span className="material-symbols-outlined text-gray-500 text-4xl">description</span>
                    </div>
                  )}

                  {/* Type Badge */}
                  <div className="absolute top-3 right-3 z-20 bg-black/40 backdrop-blur-md border border-white/10 px-2.5 py-1 rounded-full flex items-center gap-1">
                    <span className="material-symbols-outlined text-white text-[10px]">
                      {doc.type === 'pdf' ? 'picture_as_pdf' : 'image'}
                    </span>
                    <span className="text-[10px] font-bold text-white uppercase">{doc.type}</span>
                  </div>

                  {/* Delete Button */}
                  <button
                    onClick={(e) => handleDeleteDoc(e, doc.id)}
                    className="absolute top-3 left-3 z-20 size-8 bg-red-500/80 backdrop-blur-md rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <span className="material-symbols-outlined text-white text-[14px]">delete</span>
                  </button>
                </div>

                <div className="px-1">
                  <div className="flex justify-between items-start">
                    <h3 className="font-bold text-slate-900 dark:text-white text-sm truncate pr-2 leading-tight">{doc.title}</h3>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{doc.date} • {doc.size}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        {filteredDocs.length > 0 && (
          <div className="py-8 flex flex-col items-center justify-center opacity-50 gap-2">
            <span className="material-symbols-outlined text-3xl text-gray-500">check_circle</span>
            <p className="text-xs text-gray-500 font-medium">You're all caught up</p>
          </div>
        )}
      </main>

      {/* FAB */}
      <div className="absolute bottom-24 right-5 z-40">
        <button onClick={onScan} className="group relative flex size-16 items-center justify-center rounded-full bg-gradient-to-tr from-primary to-indigo-500 text-white shadow-glow hover:scale-105 active:scale-95 transition-all duration-300">
          <span className="material-symbols-outlined text-[32px] group-hover:rotate-90 transition-transform duration-300">add</span>
        </button>
      </div>
    </div>
  );
};