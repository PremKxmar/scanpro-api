import React, { useState } from 'react';

interface EditScreenProps {
  onDone: () => void;
  onBack: () => void;
}

export const EditScreen: React.FC<EditScreenProps> = ({ onDone, onBack }) => {
  const [selectedFilter, setSelectedFilter] = useState('Auto');
  const filters = [
    { name: 'Auto', color: 'bg-blue-100', img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDb6MOJsWXlAQdf7oRW0Sgjo5G-0CqOfn9Rm8ujk4_XorjX2IyEqhiBnqAOKcTeRIAa5Q4VXCaNnGs6zqkPlvmVljD6nFBeVGyYTBGm4ZTQWcAHXPnsHvbWqNqCaHQ0vPrWrhhJiUTb6G7Qvj3wxZEcmXfj8sB0Ftg8tdGuohtaywjKW0W0gT0aJCXFnTDgxmYxLsIuxiWwvbgWrXKv7RiLMABjh2cK5tRIn5XHGA2jyc3LBv8QkF2V7gsggQK24oQ9NKeekP34h8DL' },
    { name: 'B&W', color: 'bg-gray-800', img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuB0p3WtbeNU42gfEoZchWiwn6dKsq5mmdEnrpifLt6lQ5QEg2KLrxaWVIfZo1QIpEzUGeWM_XyI96_0YmlT6M6QTFuF6l1bxUciggtxxYX8C8x0ZHWXcT1rDu52QFJ2o8S5Iom_5m4PzWHzwsNHUvoz4X5qp21Q5tXzvMctlIgU9FRcYuFv9RZmq5tCo1_-IkM3q9voCOeGBzIG3mOq2P7BDk3EEJcjSuULyIYbOqnOUyRgbBOQEkD3hgvZtmfwZs8DF52-buIhabGk' },
    { name: 'Magic', color: 'bg-purple-500', img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBXQNDHBQ4yITOKgZz0wHOw1lZ7BVNjJt0Gc3Q-67t86OHK2BzGbuvrj8puSocko_OLH5vm6-r-sjsIyn0h23oMUeFMvKV_IqewhegBB64yvYg62uisW9MvpOEl1Oc3sVYJpYDnOdGTpQdghrmckSnNpEGqFNyeA3EwfTidupDLSGHxhnloob5gE_dk_ii59jg1eZR_JxCBhW1X3_Rh1mcqitNHqYKlczwz5cOGVNRTUKNRvpfUuy98aZ3xlD9Eu21X5hMxNHbzaSI7' },
    { name: 'Gray', color: 'bg-gray-400', img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDb6MOJsWXlAQdf7oRW0Sgjo5G-0CqOfn9Rm8ujk4_XorjX2IyEqhiBnqAOKcTeRIAa5Q4VXCaNnGs6zqkPlvmVljD6nFBeVGyYTBGm4ZTQWcAHXPnsHvbWqNqCaHQ0vPrWrhhJiUTb6G7Qvj3wxZEcmXfj8sB0Ftg8tdGuohtaywjKW0W0gT0aJCXFnTDgxmYxLsIuxiWwvbgWrXKv7RiLMABjh2cK5tRIn5XHGA2jyc3LBv8QkF2V7gsggQK24oQ9NKeekP34h8DL' }, // Reusing for demo
    { name: 'Light', color: 'bg-yellow-100', img: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDb6MOJsWXlAQdf7oRW0Sgjo5G-0CqOfn9Rm8ujk4_XorjX2IyEqhiBnqAOKcTeRIAa5Q4VXCaNnGs6zqkPlvmVljD6nFBeVGyYTBGm4ZTQWcAHXPnsHvbWqNqCaHQ0vPrWrhhJiUTb6G7Qvj3wxZEcmXfj8sB0Ftg8tdGuohtaywjKW0W0gT0aJCXFnTDgxmYxLsIuxiWwvbgWrXKv7RiLMABjh2cK5tRIn5XHGA2jyc3LBv8QkF2V7gsggQK24oQ9NKeekP34h8DL' },
  ];

  return (
    <div className="bg-background-dark font-display antialiased overflow-hidden h-screen w-full flex flex-col text-white select-none relative">
      
      {/* Top Bar */}
      <div className="flex items-center justify-between px-5 pt-12 pb-4 w-full z-50">
        <button onClick={onBack} className="flex size-10 items-center justify-center rounded-full bg-surface-dark border border-white/5 active:bg-white/10 transition-colors">
            <span className="material-symbols-outlined text-white text-[24px]">arrow_back_ios_new</span>
        </button>
        <h2 className="text-white text-[17px] font-semibold tracking-wide">Edit</h2>
        <button onClick={onDone} className="h-9 px-6 flex items-center justify-center rounded-full bg-primary hover:bg-primary-dark shadow-lg active:scale-95 transition-all">
            <span className="text-white text-[14px] font-semibold">Done</span>
        </button>
      </div>

      {/* Main Image Area */}
      <div className="flex-1 relative w-full flex flex-col justify-center items-center px-8 pb-32 z-10">
        <div className="relative w-full max-w-sm aspect-[3/4] group/crop">
            {/* Image */}
            <img 
              className="w-full h-full object-cover rounded-2xl shadow-2xl opacity-90" 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuDY_4_t-6GqmQu9dzWKjJ-1uus9S8_z5Tozo_nD0QJjf__lHMIoufpgQUbOdiqDrw1L6cycCMivBplI4trvlqmVHNqkz44l59C_iajR83g_BmpDkAWflaYc1ZeX9PbsY_U172vky45e3o3huss0eMlLhD_DXIvWeVKvY38mpbhZ0UbDihQSrFvam87ZmSLhKBIMinyZ43hLuqYBZlkfKx_2myk21-lpXtItob_6tEH6dVB1BjYyprMnq2VI-Y36K3fgJJXwOShxx1Ik" 
              alt="scan" 
            />
            <div className="absolute inset-0 bg-black/50 rounded-2xl pointer-events-none"></div>

            {/* Crop Overlay */}
            <div className="absolute top-[8%] left-[10%] right-[10%] bottom-[15%] z-20">
                {/* Viewport */}
                <div className="absolute inset-0 overflow-hidden rounded-sm ring-1 ring-primary/50 shadow-[0_0_0_1000px_rgba(0,0,0,0.6)]">
                     <img className="absolute w-[125%] h-[129%] max-w-none object-cover -top-[10%] -left-[12.5%]" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCSaYhVcVU3ivga3MXwdynwLzmYAOcvMO0gUWeuAmlF1o2NALK5MMdMS1ybAJncIKW0AoYmN2pCcyOCYtMJ2AkBeMXT0q_Eg1RRrpv38-C7wGB8HZ9QaMLYBVBKuw6ijCNfQHTOJQVSW_hVY4_H8-29HnXjiAOMeoc1Vjx8LJPqPjiDYJzt0HmVzX3DHwHI6B5DfA28FQ-qX9qkdCrjXQOzzJPtbjr5j2V5efQsn5ypH139WnJtWjMBRJHu2kbln2D7v6ciW_AQdm48" alt="crop view" />
                     {/* Grid */}
                     <div className="absolute inset-0 flex flex-col justify-between pointer-events-none opacity-30">
                        <div className="flex-1 border-b border-white/50"></div>
                        <div className="flex-1 border-b border-white/50"></div>
                        <div className="flex-1"></div>
                    </div>
                    <div className="absolute inset-0 flex flex-row justify-between pointer-events-none opacity-30">
                        <div className="flex-1 border-r border-white/50"></div>
                        <div className="flex-1 border-r border-white/50"></div>
                        <div className="flex-1"></div>
                    </div>
                </div>

                {/* Handles - styled like screenshot */}
                <div className="absolute -top-3 -left-3 size-6 bg-white rounded-full shadow-lg flex items-center justify-center z-30 touch-none">
                    <div className="size-2.5 bg-primary rounded-full"></div>
                </div>
                <div className="absolute -top-3 -right-3 size-6 bg-white rounded-full shadow-lg flex items-center justify-center z-30 touch-none">
                     <div className="size-2.5 bg-primary rounded-full"></div>
                </div>
                <div className="absolute -bottom-3 -left-3 size-6 bg-white rounded-full shadow-lg flex items-center justify-center z-30 touch-none">
                     <div className="size-2.5 bg-primary rounded-full"></div>
                </div>
                <div className="absolute -bottom-3 -right-3 size-6 bg-white rounded-full shadow-lg flex items-center justify-center z-30 touch-none">
                     <div className="size-2.5 bg-primary rounded-full"></div>
                </div>
                
                {/* Magnifier on active (simulated) */}
                <div className="absolute -top-24 -left-10 size-24 rounded-full border-[3px] border-white overflow-hidden shadow-2xl z-50 hidden group-active/crop:block animate-fadeIn">
                     <img className="w-[300%] max-w-none absolute top-[-50px] left-[-50px]" src="https://lh3.googleusercontent.com/aida-public/AB6AXuCEhRz5A3FLTWkjkUD82-404B8NUkvj2Iqfqdkyh3Zru8VOv3kdNaNf3907oeeoYn78_K1VT_I2nq3fw5vlNxSeCjzmLxenQ6oTseCZg0bUfK1-DZbSsCOgaXlz_hI9oLJDkBAoPEIkfbQl6lJfDSop-uDnLW7yPU-h_vFqsGs3D3OThogBbMKlQhaJf1WVx-R8fQsv5GszNlBD8tWhwIp7X8Tu-_ByTt8wx1PGMBq8evVeabXp8QcPE6-lji0ucLxHzusfaEMZ38Li" alt="magnifier" />
                     <div className="absolute inset-0 flex items-center justify-center">
                        <span className="material-symbols-outlined text-primary drop-shadow-md text-3xl">add</span>
                     </div>
                 </div>
            </div>
        </div>
      </div>

      {/* Bottom Controls Area */}
      <div className="absolute bottom-0 left-0 right-0 z-50 pb-8 flex flex-col gap-6">
         
         {/* Filters Scroll */}
         <div className="w-full overflow-x-auto no-scrollbar pl-6">
            <div className="flex items-center gap-6 min-w-max pr-6">
                {filters.map((filter) => (
                    <button 
                        key={filter.name}
                        onClick={() => setSelectedFilter(filter.name)}
                        className="flex flex-col items-center gap-2 group"
                    >
                        <div className={`size-[3.75rem] rounded-full p-0.5 relative transition-all duration-200 ${selectedFilter === filter.name ? 'ring-2 ring-primary ring-offset-2 ring-offset-background-dark' : 'opacity-70 hover:opacity-100'}`}>
                            <img className="w-full h-full object-cover rounded-full" src={filter.img} alt={filter.name} />
                             {filter.name === 'Auto' && selectedFilter === 'Auto' && (
                                 <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 bg-primary text-white text-[9px] px-1.5 py-0.5 rounded-full font-bold shadow-sm">
                                     Auto
                                 </div>
                             )}
                        </div>
                        <span className={`text-[11px] font-medium tracking-wide ${selectedFilter === filter.name ? 'text-primary' : 'text-gray-400'}`}>
                            {filter.name}
                        </span>
                    </button>
                ))}
            </div>
         </div>

         {/* Bottom Action Pill */}
         <div className="px-6 flex justify-center">
            <div className="bg-[#1e1f2e] border border-white/5 rounded-[2rem] px-8 py-3.5 flex items-center gap-10 shadow-xl">
                <button className="group flex flex-col items-center justify-center gap-1">
                    <div className="size-6 text-primary flex items-center justify-center">
                        <span className="material-symbols-outlined text-[24px]">crop</span>
                    </div>
                     <div className="h-1 w-1 bg-primary rounded-full mt-1"></div>
                </button>
                <button className="group flex flex-col items-center justify-center gap-1 opacity-50 hover:opacity-100 transition">
                    <div className="size-6 text-white flex items-center justify-center">
                        <span className="material-symbols-outlined text-[24px]">rotate_right</span>
                    </div>
                     <div className="h-1 w-1 bg-transparent rounded-full mt-1"></div>
                </button>
                <button className="group flex flex-col items-center justify-center gap-1 opacity-50 hover:opacity-100 transition">
                    <div className="size-6 text-white flex items-center justify-center">
                        <span className="material-symbols-outlined text-[24px]">tune</span>
                    </div>
                     <div className="h-1 w-1 bg-transparent rounded-full mt-1"></div>
                </button>
                 <button className="group flex flex-col items-center justify-center gap-1 opacity-50 hover:opacity-100 transition">
                    <div className="size-6 text-white flex items-center justify-center">
                        <span className="material-symbols-outlined text-[24px]">auto_fix_high</span>
                    </div>
                     <div className="h-1 w-1 bg-transparent rounded-full mt-1"></div>
                </button>
            </div>
         </div>
      </div>
    </div>
  );
};
