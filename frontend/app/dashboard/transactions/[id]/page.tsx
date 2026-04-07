'use client';

import {
    ShieldAlert,
    ArrowLeft,
    CheckCircle2,
    XCircle,
    Zap,
    BrainCircuit,
    Search,
    Download,
    Share2,
    Calendar,
    MessageSquareMore,
    CreditCard,
    MapPin
} from 'lucide-react';
import Link from 'next/link';
import { toast } from 'sonner';

import { useRole } from '@/app/context/RoleContext';

export default function TransactionDetailPage({ params }: { params: { id: string } }) {
    const { role } = useRole();
    const handleAction = (action: string) => {
        toast.info(`Executing analytical action: ${action}`);
    };

    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {/* Context Header */}
            <div className="flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    <Link href="/dashboard" className="p-2 hover:bg-white rounded-xl border border-transparent hover:border-slate-200 transition-all text-slate-400 hover:text-slate-900 shrink-0">
                        <ArrowLeft className="w-5 h-5" />
                    </Link>
                    <div>
                        <div className="flex items-center gap-2 mb-0.5">
                            <span className="bg-slate-100 text-slate-600 text-[10px] font-black px-2 py-0.5 rounded-full uppercase tracking-widest border border-slate-200">
                                TXN-ID: {params.id || 'GH-99823'}
                            </span>
                            <span className="bg-amber-100 text-amber-700 text-[10px] font-black px-2 py-0.5 rounded-full uppercase tracking-widest border border-amber-200">
                                Pending {role?.replace('_', ' ').toUpperCase()} Review
                            </span>
                        </div>
                        <h1 className="text-2xl font-black text-slate-900 tracking-tight flex items-center gap-3">
                            International Wire Transfer - $12,500.00
                        </h1>
                    </div>
                </div>
                <div className="flex items-center gap-3">
                    <button className="p-2.5 text-slate-500 hover:text-slate-900 hover:bg-white rounded-xl transition-all border border-transparent hover:border-slate-200">
                        <Share2 className="w-5 h-5" />
                    </button>
                    <button className="p-2.5 text-slate-500 hover:text-slate-900 hover:bg-white rounded-xl transition-all border border-transparent hover:border-slate-200">
                        <Download className="w-5 h-5" />
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Main Content Area */}
                <div className="lg:col-span-3 space-y-6">
                    {/* Core Stats Bar */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-1 px-1">
                        <div className="bg-white rounded-2xl border border-slate-200 p-4 flex items-center gap-4">
                            <div className="w-10 h-10 rounded-xl bg-slate-50 flex items-center justify-center text-slate-400">
                                <Calendar className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Transaction Date</p>
                                <p className="text-sm font-bold text-slate-900">Oct 24, 2023 • 14:32</p>
                            </div>
                        </div>
                        <div className="bg-white rounded-2xl border border-slate-200 p-4 flex items-center gap-4">
                            <div className="w-10 h-10 rounded-xl bg-slate-50 flex items-center justify-center text-slate-400">
                                <CreditCard className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Payment Method</p>
                                <p className="text-sm font-bold text-slate-900">SWIFT / International</p>
                            </div>
                        </div>
                        <div className="bg-white rounded-2xl border border-slate-200 p-4 flex items-center gap-4">
                            <div className="w-10 h-10 rounded-xl bg-slate-50 flex items-center justify-center text-slate-400">
                                <MapPin className="w-5 h-5" />
                            </div>
                            <div>
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Origin Terminal</p>
                                <p className="text-sm font-bold text-slate-900">Node Accra-Terminal-22</p>
                            </div>
                        </div>
                    </div>

                    {/* AI Risk Analysis */}
                    <div className="bg-white rounded-3xl border border-slate-200 shadow-xl shadow-slate-200/50 overflow-hidden">
                        <div className="p-6 border-b border-slate-100 flex items-center justify-between">
                            <div className="absolute top-0 right-0 p-3 text-slate-400">
                                <ShieldAlert className="w-12 h-12 opacity-5 -rotate-90" />
                            </div>
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-blue-50 rounded-xl text-blue-600">
                                    <BrainCircuit className="w-6 h-6" />
                                </div>
                                <div>
                                    <h3 className="font-black text-lg text-slate-900">AI Risk Analysis</h3>
                                    <p className="text-xs text-slate-400 font-bold">GENERATIVE NEURAL EXPLANATION</p>
                                </div>
                            </div>
                        </div>
                        <div className="p-8">
                            <div className="prose prose-slate max-w-none">
                                <p className="text-slate-600 font-medium leading-relaxed">
                                    This transaction has been flagged as <span className="text-red-600 font-bold underline decoration-red-200 transition-all">High Risk</span> due to a convergence of anomalous vectors.
                                    The requested amount of <span className="font-bold text-slate-900">$12,500.00</span> exceeds the user&apos;s average daily transfer limit by <span className="text-amber-600 font-bold">450%</span>.
                                    Furthermore, the device fingerprint indicates a new operating system accessed from an IP address in <span className="font-bold text-slate-800 underline decoration-slate-200">Lagos, Nigeria</span>,
                                    while the user&apos;s last known legitimate login was only 2 hours ago in <span className="font-bold text-slate-800 underline decoration-slate-200 transition-all">Accra, Ghana</span>.
                                    This &quot;impossible travel&quot; velocity is a primary indicator of account takeover.
                                </p>
                            </div>

                            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
                                {[
                                    { label: 'Impossible Travel', score: 'Flagged', color: 'text-red-500', bg: 'bg-red-50' },
                                    { label: 'Amount Variance (+450%)', score: 'Flagged', color: 'text-amber-500', bg: 'bg-amber-50' },
                                    { label: 'New Device ID', score: 'Flagged', color: 'text-blue-500', bg: 'bg-blue-50' }
                                ].map((badge, i) => (
                                    <div key={i} className={`p-4 rounded-2xl ${badge.bg} border-2 border-white flex flex-col items-center text-center shadow-sm`}>
                                        <ShieldAlert className={`w-5 h-5 mb-2 ${badge.color}`} />
                                        <p className="text-xs font-black text-slate-900 mb-0.5">{badge.label}</p>
                                        <p className={`text-[10px] font-black uppercase tracking-widest ${badge.color}`}>{badge.score}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Risk Factor Attribution */}
                    <div className="bg-white rounded-3xl border border-slate-200 shadow-sm overflow-hidden p-6">
                        <div className="flex items-center justify-between mb-8">
                            <h3 className="font-black text-lg text-slate-900">Risk Factor Attribution</h3>
                            <button className="text-xs font-black text-blue-600 hover:underline">VIEW FULL FEATURE LIST</button>
                        </div>
                        <div className="space-y-6">
                            {[
                                { factor: 'Geoloc Velocity (Impossible Travel)', impact: '+42% Impact', value: 85, color: 'bg-red-500' },
                                { factor: 'Transaction Amount Deviation', impact: '+28% Impact', value: 65, color: 'bg-orange-500' },
                                { factor: 'Beneficiary Age (New Account)', impact: '+15% Impact', value: 35, color: 'bg-blue-400' },
                                { factor: 'Device Fingerprint Mismatch', impact: '+12% Impact', value: 25, color: 'bg-indigo-300' }
                            ].map((item, i) => (
                                <div key={i} className="space-y-2">
                                    <div className="flex justify-between items-end text-xs font-bold">
                                        <span className="text-slate-600">{item.factor}</span>
                                        <span className="text-red-600 font-black">{item.impact}</span>
                                    </div>
                                    <div className="h-2 w-full bg-slate-50 rounded-full overflow-hidden">
                                        <div className={`h-full ${item.color} rounded-r-sm`} style={{ width: `${item.value}%` }} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Transaction History */}
                    <div className="bg-white rounded-3xl border border-slate-200 overflow-hidden shadow-sm">
                        <div className="p-6 border-b border-slate-100 flex items-center justify-between">
                            <h3 className="font-black text-lg text-slate-900">Related Activity History</h3>
                            <button className="text-slate-400 hover:text-slate-900 transition-colors">
                                <Search className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left">
                                <thead className="bg-slate-50 text-[10px] font-black text-slate-500 uppercase tracking-widest font-mono">
                                    <tr>
                                        <th className="px-6 py-4">Status</th>
                                        <th className="px-6 py-4">Merchant / Counterparty</th>
                                        <th className="px-6 py-4">Location</th>
                                        <th className="px-6 py-4">Amount</th>
                                        <th className="px-6 py-4 text-right">Result</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-50">
                                    {[
                                        { status: 'CURRENT', entity: 'Global Wire Transfer', loc: 'Lagos, NG', amt: '$12,500.00', result: 'FLAGGED', color: 'text-red-600 bg-red-100 border-red-200' },
                                        { status: '2h ago', entity: 'Uber Rides', loc: 'Accra, GH', amt: '$14.50', result: 'CLEARED', color: 'text-emerald-600 bg-emerald-100 border-emerald-200' },
                                        { status: '1d ago', entity: 'Shoprite Groceries', loc: 'Accra, GH', amt: '$85.20', result: 'CLEARED', color: 'text-emerald-600 bg-emerald-100 border-emerald-200' },
                                        { status: '2d ago', entity: 'Vodafone Topup', loc: 'Accra, GH', amt: '$10.00', result: 'CLEARED', color: 'text-emerald-600 bg-emerald-100 border-emerald-200' },
                                    ].map((row, i) => (
                                        <tr key={i} className="hover:bg-slate-50 transition-colors">
                                            <td className="px-6 py-4">
                                                <span className={`px-2 py-0.5 rounded text-[10px] font-black border uppercase tracking-wider ${row.color}`}>
                                                    {row.status}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 font-bold text-slate-900">{row.entity}</td>
                                            <td className="px-6 py-4 text-slate-500 font-medium">{row.loc}</td>
                                            <td className="px-6 py-4 font-mono font-bold text-slate-900">{row.amt}</td>
                                            <td className="px-6 py-4 text-right">
                                                <div className="flex items-center justify-end gap-1.5 font-black text-xs text-slate-400">
                                                    {row.result === 'CLEARED' ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> : <ShieldAlert className="w-4 h-4 text-red-500" />}
                                                    {row.result}
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Analytical Sidebar Controls */}
                <div className="space-y-6">
                    {/* Risk Score Widget */}
                    <div className="bg-white rounded-3xl border border-slate-200 p-8 flex flex-col items-center text-center shadow-sm relative overflow-hidden group">
                        <div className="absolute top-0 left-0 w-full h-1 bg-red-500" />
                        <div className="w-32 h-32 relative mb-6">
                            <svg className="w-full h-full -rotate-90">
                                <circle cx="64" cy="64" r="58" fill="none" stroke="#f1f5f9" strokeWidth="12" />
                                <circle
                                    cx="64" cy="64" r="58" fill="none" stroke="#ef4444" strokeWidth="12"
                                    strokeDasharray="364.4" strokeDashoffset="40" strokeLinecap="round"
                                />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
                                <span className="text-4xl font-black text-slate-900 leading-none">89</span>
                                <span className="text-[10px] font-black text-red-600 uppercase tracking-widest mt-1">Critical</span>
                            </div>
                        </div>
                        <h4 className="font-black text-slate-900 text-lg leading-tight mb-2">Fraud Probability</h4>
                        <p className="text-xs text-slate-500 font-bold leading-relaxed mb-6">
                            High-confidence signal indicating non-human interaction patterns during authorization.
                        </p>
                        <div className="grid grid-cols-2 w-full gap-4 text-left border-t border-slate-100 pt-6">
                            <div>
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">THRESHOLD</p>
                                <p className="text-sm font-bold text-slate-900">75+</p>
                            </div>
                            <div>
                                <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">MODEL VER</p>
                                <p className="text-sm font-bold text-slate-900">v-Alpha.9</p>
                            </div>
                        </div>
                    </div>

                    {/* Action Panel */}
                    <div className="bg-white rounded-3xl border border-slate-200 p-6 space-y-4 shadow-sm">
                        <h4 className="font-black text-slate-900 text-sm uppercase tracking-widest mb-2 flex items-center gap-2">
                            <Zap className="w-4 h-4 text-amber-500" /> {role?.replace('_', ' ').toUpperCase()} Action Panel
                        </h4>

                        <div className="space-y-3">
                            {role === 'senior_analyst' && (
                                <>
                                    <button
                                        onClick={() => handleAction('APPROVE')}
                                        className="w-full py-4 bg-emerald-600 text-white rounded-2xl font-black text-sm flex items-center justify-center gap-2 hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-200"
                                    >
                                        <CheckCircle2 className="w-5 h-5" /> Approve Transaction
                                    </button>
                                    <button
                                        onClick={() => handleAction('BLOCK')}
                                        className="w-full py-4 bg-red-600 text-white rounded-2xl font-black text-sm flex items-center justify-center gap-2 hover:bg-red-700 transition-all shadow-lg shadow-red-200"
                                    >
                                        <XCircle className="w-5 h-5" /> Block & Flag Account
                                    </button>
                                </>
                            )}
                            <button
                                onClick={() => handleAction('MFA')}
                                className="w-full py-4 border-2 border-slate-200 text-slate-700 rounded-2xl font-black text-sm flex items-center justify-center gap-2 hover:bg-slate-50 transition-all"
                            >
                                <ShieldAlert className="w-5 h-5" /> {role === 'junior_analyst' ? 'Request Step-Up Auth' : 'Trigger Step-Up Auth'}
                            </button>
                            {role === 'junior_analyst' && (
                                <button
                                    onClick={() => handleAction('ELEVATE')}
                                    className="w-full py-4 bg-slate-900 text-white rounded-2xl font-black text-sm flex items-center justify-center gap-2 hover:bg-slate-800 transition-all shadow-lg"
                                >
                                    <Share2 className="w-5 h-5" /> Escalate to Senior
                                </button>
                            )}
                        </div>

                        <div className="pt-2">
                            <button className="w-full py-3 text-slate-400 hover:text-slate-600 text-xs font-bold transition-colors flex items-center justify-center gap-2">
                                <MessageSquareMore className="w-4 h-4" /> Add Investigation Note
                            </button>
                        </div>
                    </div>

                    {/* Risk Level Legend */}
                    <div className="bg-slate-900 rounded-3xl p-6 text-white shadow-xl">
                        <h4 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Risk Categorization</h4>
                        <div className="space-y-4">
                            {[
                                { level: 'Critical', score: '85-100', color: 'bg-red-500' },
                                { level: 'High Risk', score: '60-84', color: 'bg-orange-500' },
                                { level: 'Medium', score: '30-59', color: 'bg-amber-500' },
                                { level: 'Low Risk', score: '0-29', color: 'bg-emerald-500' }
                            ].map((row, i) => (
                                <div key={i} className="flex items-center justify-between group cursor-default">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-2 h-2 rounded-full ${row.color}`} />
                                        <span className="text-xs font-bold group-hover:translate-x-1 transition-transform">{row.level}</span>
                                    </div>
                                    <span className="text-[10px] font-mono text-slate-500 font-bold">{row.score}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
