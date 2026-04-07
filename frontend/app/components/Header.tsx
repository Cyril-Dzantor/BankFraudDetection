'use client';

import { Search, Plus, Bell, HelpCircle } from 'lucide-react';
import { toast } from 'sonner';

export default function Header() {
    return (
        <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-6 sticky top-0 z-10 w-full">
            <div className="flex items-center gap-6 flex-1">
                <div className="flex items-center gap-2 bg-emerald-50 border border-emerald-100 px-3 py-1.5 rounded-full">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-xs font-bold text-emerald-700 uppercase tracking-widest">System Health: Normal</span>
                </div>

                <div className="max-w-md w-full relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search accounts, transaction IDs, or alerts..."
                        className="w-full pl-9 pr-4 py-2 bg-slate-100 border-none rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none placeholder-slate-400"
                    />
                </div>
            </div>

            <div className="flex items-center gap-4">
                <button
                    onClick={() => toast.success('New case form launched')}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 shadow-sm transition-colors"
                >
                    <Plus className="w-4 h-4" />
                    New Case
                </button>
                <div className="w-px h-6 bg-slate-200" />
                <button
                    onClick={() => toast.info('You have 3 unread high-priority alerts')}
                    className="text-slate-500 hover:text-slate-900 transition-colors relative"
                >
                    <Bell className="w-5 h-5" />
                    <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full border border-white" />
                </button>
                <button
                    onClick={() => toast('Help center opened')}
                    className="text-slate-500 hover:text-slate-900 transition-colors"
                >
                    <HelpCircle className="w-5 h-5" />
                </button>
            </div>
        </header>
    );
}
