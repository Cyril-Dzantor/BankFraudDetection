'use client';

import {
    Users,
    Search,
    Filter,
    ArrowRight,
    MapPin,
    ShieldCheck,
    ShieldAlert,
    ShieldX,
    UserCircle2
} from 'lucide-react';
import { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import { toast } from 'sonner';
import { getBackendUrl } from '@/app/utils/api';

export default function AccountsDirectory() {
    const [accounts, setAccounts] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');

    const backendUrl = getBackendUrl();

    useEffect(() => {
        const fetchAccounts = async () => {
            try {
                const res = await fetch(`${backendUrl}/api/v1/accounts/`);
                if (!res.ok) throw new Error('Failed to fetch accounts');
                const data = await res.json();
                setAccounts(data);
            } catch (error) {
                console.error(error);
                toast.error('Could not load account directory');
            } finally {
                setIsLoading(false);
            }
        };
        fetchAccounts();
    }, [backendUrl]);

    const filteredAccounts = useMemo(() => {
        return accounts.filter(acc =>
            acc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            acc.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
            acc.location.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }, [accounts, searchTerm]);

    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {/* Header */}
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Customer Directory</h1>
                    <p className="text-slate-500 text-sm mt-1">Global registry of institutional and retail accounts for forensic discovery.</p>
                </div>
                <div className="flex items-center gap-3">
                    <span className="px-3 py-1 bg-blue-50 text-blue-600 rounded-full text-xs font-bold border border-blue-100">
                        {accounts.length} Entities Registered
                    </span>
                </div>
            </div>

            {/* Search & Filter */}
            <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-sm flex items-center gap-4">
                <div className="flex-1 relative">
                    <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search by name, account number, or location..."
                        className="w-full pl-9 pr-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-all font-medium"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
                <button className="flex items-center gap-2 px-4 py-2.5 bg-white border border-slate-200 rounded-xl text-sm font-bold text-slate-700 hover:bg-slate-50 transition-colors">
                    <Filter className="w-4 h-4" /> Advanced Filters
                </button>
            </div>

            {/* Account Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {isLoading ? (
                    Array(6).fill(0).map((_, i) => (
                        <div key={i} className="h-48 bg-white border border-slate-200 rounded-2xl animate-pulse" />
                    ))
                ) : filteredAccounts.length > 0 ? (
                    filteredAccounts.map((acc) => (
                        <Link
                            key={acc.id}
                            href={`/dashboard/accounts/${acc.id}`}
                            className="group bg-white border border-slate-200 rounded-2xl p-6 hover:border-blue-300 hover:shadow-xl hover:shadow-blue-500/5 transition-all duration-300 flex flex-col"
                        >
                            <div className="flex justify-between items-start mb-4">
                                <div className="w-12 h-12 bg-slate-50 rounded-xl flex items-center justify-center text-slate-400 border border-slate-100 group-hover:bg-blue-50 group-hover:text-blue-600 group-hover:border-blue-100 transition-colors">
                                    {acc.initials ? (
                                        <span className="font-black text-xs">{acc.initials}</span>
                                    ) : (
                                        <UserCircle2 className="w-6 h-6" />
                                    )}
                                </div>
                                <div className={`px-2.5 py-1 rounded-lg text-[9px] font-black uppercase tracking-widest border
                                    ${acc.risk_level === 'Low Risk' ? 'bg-emerald-50 text-emerald-700 border-emerald-100' :
                                        acc.risk_level === 'Critical' ? 'bg-red-50 text-red-700 border-red-100' :
                                            'bg-amber-50 text-amber-700 border-amber-100'}`}>
                                    {acc.risk_level}
                                </div>
                            </div>

                            <div className="flex-1">
                                <h3 className="font-black text-slate-900 group-hover:text-blue-700 transition-colors">{acc.name}</h3>
                                <p className="text-xs font-mono text-slate-400 mt-0.5">{acc.id}</p>

                                <div className="mt-4 space-y-2">
                                    <div className="flex items-center gap-2 text-[10px] text-slate-500 font-bold uppercase tracking-wide">
                                        <MapPin className="w-3 h-3" /> {acc.location}
                                    </div>
                                    <div className="flex items-center gap-2 text-[10px] text-slate-500 font-bold uppercase tracking-wide">
                                        {acc.account_status === 'ACTIVE' ? (
                                            <>
                                                <ShieldCheck className="w-3 h-3 text-emerald-500" />
                                                <span className="text-emerald-600">Active Status</span>
                                            </>
                                        ) : acc.account_status === 'FROZEN' ? (
                                            <>
                                                <ShieldX className="w-3 h-3 text-red-500" />
                                                <span className="text-red-600">Frozen Restricted</span>
                                            </>
                                        ) : (
                                            <>
                                                <ShieldAlert className="w-3 h-3 text-amber-500" />
                                                <span className="text-amber-600">Under Review</span>
                                            </>
                                        )}
                                    </div>
                                </div>
                            </div>

                            <div className="mt-6 pt-4 border-t border-slate-50 flex items-center justify-between">
                                <div className="flex flex-col">
                                    <span className="text-[9px] font-black text-slate-400 uppercase tracking-widest">Neural Risk</span>
                                    <span className="text-sm font-black text-slate-900">{(acc.risk_score * 100).toFixed(0)}/100</span>
                                </div>
                                <ArrowRight className="w-5 h-5 text-slate-300 group-hover:text-blue-600 group-hover:translate-x-1 transition-all" />
                            </div>
                        </Link>
                    ))
                ) : (
                    <div className="col-span-full py-12 text-center">
                        <Users className="w-12 h-12 text-slate-200 mx-auto mb-4" />
                        <p className="text-slate-500 font-bold">No accounts found matching your search.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
