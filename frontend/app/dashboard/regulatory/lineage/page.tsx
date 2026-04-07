'use client';

import {
    History,
    Search,
    Calendar,
    ChevronDown,
    Filter,
    Download,
    Printer,
    CheckCircle2,
    XCircle,
    Clock,
    User,
    ShieldCheck,
    Lock,
    ExternalLink
} from 'lucide-react';
import { toast } from 'sonner';
import { useRole } from '@/app/context/RoleContext';

export default function DecisionLineagePage() {
    const { role } = useRole();
    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-0.5 flex items-center gap-2">
                        <Lock className="w-3 h-3" /> {role?.replace('_', ' ').toUpperCase()} ACCESS MODE
                    </p>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Decision Lineage Viewer</h1>
                    <p className="text-slate-500 text-sm mt-1 max-w-2xl font-medium">
                        Search and review transaction decision histories for your {role === 'system_admin' ? 'organization' : 'assigned cases'}. {role === 'junior_analyst' ? 'Full audit logs and compliance summaries are hidden in this view.' : 'System-wide compliance summaries are available for oversight.'}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button className="flex items-center gap-2 px-4 py-2 border border-slate-200 bg-white rounded-xl text-sm font-bold text-slate-600 shadow-sm hover:bg-slate-50 transition-all">
                        <Printer className="w-4 h-4" /> Print PDF
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-xl text-sm font-bold shadow-lg shadow-slate-200 hover:bg-slate-800 transition-all">
                        <Download className="w-4 h-4" /> Export Summary
                    </button>
                </div>
            </div>

            {/* Global Search Bar */}
            <div className="bg-white p-6 rounded-3xl border border-slate-200 shadow-xl shadow-slate-200/40 flex flex-col md:flex-row gap-4 items-end">
                <div className="flex-1 space-y-2 w-full">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest pl-1">Search ID or Case #</label>
                    <div className="relative">
                        <Search className="w-5 h-5 absolute left-4 top-1/2 -translate-y-1/2 text-slate-300" />
                        <input
                            type="text"
                            placeholder="e.g. TRX-2023-8849-GH"
                            className="w-full pl-12 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-2xl text-sm font-bold focus:bg-white focus:ring-2 focus:ring-slate-900/5 focus:border-slate-900 outline-none transition-all placeholder:text-slate-300"
                        />
                    </div>
                </div>
                <div className="w-full md:w-48 space-y-2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest pl-1">Timeframe</label>
                    <button className="w-full flex items-center justify-between px-4 py-3 bg-white border border-slate-200 rounded-2xl text-sm font-bold text-slate-700 hover:bg-slate-50 transition-colors">
                        <span className="flex items-center gap-2"><Calendar className="w-4 h-4" /> Last 24h</span>
                        <ChevronDown className="w-4 h-4" />
                    </button>
                </div>
                <div className="w-full md:w-48 space-y-2">
                    <label className="text-[10px] font-black text-slate-400 uppercase tracking-widest pl-1">Verdict Type</label>
                    <button className="w-full flex items-center justify-between px-4 py-3 bg-white border border-slate-200 rounded-2xl text-sm font-bold text-slate-700 hover:bg-slate-50 transition-colors">
                        <span>All Decisions</span>
                        <ChevronDown className="w-4 h-4" />
                    </button>
                </div>
                <button className="bg-blue-600 text-white px-8 py-3.5 rounded-2xl font-black text-sm shadow-lg shadow-blue-100 hover:bg-blue-700 transition-all active:scale-95">
                    Search
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Case List */}
                <div className="lg:col-span-2 space-y-4">
                    <div className="flex items-center justify-between px-2 mb-2">
                        <h3 className="font-black text-slate-900 text-lg uppercase tracking-tight">Recent Assigned Cases</h3>
                        <span className="text-xs font-bold text-slate-400">Showing 4 of 12 assigned</span>
                    </div>

                    {[
                        { id: '#TRX - 9932 - GH', severity: 'HIGH RISK', title: 'International Wire Transfer - $45,000', origin: 'Accra Main Branch', destination: 'Offshore Holdings Ltd.', reason: 'Velocity Limit', time: '2 mins ago', color: 'border-red-500' },
                        { id: '#TRX - 8821 - GH', severity: 'MEDIUM RISK', title: 'Multiple Small Withdrawals', origin: 'Kumasi ATM #04', destination: 'N/A (Cash)', reason: 'Structuring Pattern', time: '45 mins ago', color: 'border-amber-500' },
                        { id: '#TRX - 7743 - GH', severity: 'UNDER REVIEW', title: 'Corporate Account Opening', origin: 'Global Ventures Ltd.', destination: 'Business Checking', reason: 'Incomplete UBO', time: '2 hrs ago', color: 'border-slate-400' }
                    ].map((item, i) => (
                        <div key={i} className={`bg-white rounded-3xl border-l-[6px] ${item.color} border-y border-r border-slate-200 p-6 flex items-start justify-between shadow-sm hover:shadow-md transition-shadow group cursor-pointer`}>
                            <div className="space-y-4 flex-1">
                                <div className="flex items-center gap-3">
                                    <span className="font-mono text-xs font-black text-slate-400">{item.id}</span>
                                    <span className={`text-[10px] font-black px-2 py-0.5 rounded-full ${item.severity === 'HIGH RISK' ? 'bg-red-50 text-red-600' : 'bg-slate-100 text-slate-600'}`}>{item.severity}</span>
                                    <span className="text-[10px] font-bold text-slate-400 flex items-center gap-1">
                                        <Clock className="w-3 h-3" /> {item.time}
                                    </span>
                                </div>
                                <h4 className="font-black text-slate-900 text-xl tracking-tight leading-none group-hover:text-blue-600 transition-colors">{item.title}</h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <div>
                                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">ORIGIN</p>
                                        <p className="text-xs font-bold text-slate-600">{item.origin}</p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">DESTINATION</p>
                                        <p className="text-xs font-bold text-slate-600">{item.destination}</p>
                                    </div>
                                    <div>
                                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">FLAG REASON</p>
                                        <p className="text-xs font-bold text-red-600">{item.reason}</p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-2 pt-2">
                                    <div className="w-6 h-6 rounded-full bg-slate-200 border border-white flex items-center justify-center overflow-hidden">
                                        <User className="w-4 h-4 text-slate-400" />
                                    </div>
                                    <p className="text-[10px] font-bold text-slate-500">Assigned to: <span className="text-slate-900 underline decoration-slate-200">John Doe</span></p>
                                </div>
                            </div>
                            <div className="flex flex-col items-end gap-1 text-blue-600 font-black text-[10px] uppercase tracking-widest mt-1 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                                <span>View Lineage</span>
                                <ExternalLink className="w-4 h-4" />
                            </div>
                        </div>
                    ))}
                </div>

                {/* Lineage Details Sidebar */}
                <div className="space-y-6">
                    <div className="bg-white rounded-[40px] border border-slate-200 p-8 shadow-2xl shadow-slate-200/50">
                        <div className="flex items-center justify-between mb-8">
                            <h3 className="font-black text-slate-900 text-lg">Lineage Details</h3>
                            <div className="flex gap-2">
                                <button className="p-2 text-slate-400 hover:text-slate-600"><Printer className="w-4 h-4" /></button>
                                <button className="p-2 text-slate-400 hover:text-slate-600"><Download className="w-4 h-4" /></button>
                            </div>
                        </div>

                        {/* Timeline */}
                        <div className="space-y-8 relative before:absolute before:left-[11px] before:top-2 before:bottom-2 before:w-[2px] before:bg-slate-100">
                            {[
                                { time: 'Today, 10:42 AM', label: 'Transaction Initiated', sub: 'Source: Mobile Banking App (v4.2)', meta: 'Device ID: 883-AA-291', status: 'current' },
                                { time: 'Today, 10:42 AM', label: 'Rule Engine Check', status: 'warning', tags: ['KYC: PASS', 'VELOCITY: FAIL'] },
                                { time: 'Today, 10:44 AM', label: 'Manual Review Required', sub: 'Flagged due to high velocity transfer exceeding daily threshold.', status: 'active' }
                            ].map((step, i) => (
                                <div key={i} className="relative pl-10 group">
                                    <div className={`absolute left-0 top-1.5 w-6 h-6 rounded-full border-4 border-white shadow-sm flex items-center justify-center transition-transform group-hover:scale-110
                                        ${step.status === 'current' ? 'bg-emerald-500' : step.status === 'warning' ? 'bg-orange-500' : 'bg-blue-600 shadow-blue-200 ring-2 ring-blue-100'}
                                    `}>
                                        {step.status === 'current' && <CheckCircle2 className="w-2.5 h-2.5 text-white" />}
                                        {step.status === 'warning' && <XCircle className="w-2.5 h-2.5 text-white" />}
                                    </div>
                                    <div className="space-y-1">
                                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{step.time}</p>
                                        <p className="font-black text-slate-900 text-sm leading-tight">{step.label}</p>
                                        {step.sub && <p className="text-xs text-slate-500 font-medium leading-relaxed">{step.sub}</p>}
                                        {step.meta && <p className="text-[10px] font-black text-slate-400 italic font-mono">{step.meta}</p>}
                                        {step.tags && (
                                            <div className="flex gap-2 mt-2">
                                                {step.tags.map(tag => (
                                                    <span key={tag} className="text-[8px] font-black px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                                                        {tag}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}

                            {/* Role-Based Actions Box */}
                            <div className="bg-slate-50 rounded-2xl p-4 border border-blue-100 shadow-inner mt-4">
                                <p className="text-[10px] font-black text-blue-600 uppercase tracking-wider mb-3">{role?.replace('_', ' ')} Actions</p>
                                <div className="grid grid-cols-2 gap-2">
                                    <button onClick={() => toast('Case escalation initiated')} className="py-2.5 bg-white border border-slate-200 rounded-xl text-xs font-black text-slate-700 hover:bg-white hover:border-blue-200 hover:text-blue-600 transition-all shadow-sm">Escalate Case</button>
                                    <button onClick={() => toast('Information request sent to branch')} className="py-2.5 bg-white border border-slate-200 rounded-xl text-xs font-black text-slate-700 hover:bg-white hover:border-blue-200 hover:text-blue-600 transition-all shadow-sm">Request Info</button>
                                </div>
                                {role !== 'junior_analyst' && (
                                    <button onClick={() => toast.success('Flag cleared - decision logged')} className="w-full mt-2 py-3 bg-emerald-600 text-white rounded-xl text-xs font-black shadow-lg shadow-emerald-100 hover:bg-emerald-700 transition-all">Clear Flag (Approve)</button>
                                )}
                                <p className="text-center text-[8px] font-bold text-slate-400 mt-3">Action logged: ID-299381 • PERSPECTIVE: {role.toUpperCase()}</p>
                            </div>
                        </div>
                    </div>

                    {/* Restricted Data Warning */}
                    <div className="bg-slate-50 border border-slate-200 rounded-2xl p-6 flex gap-4">
                        <History className="w-6 h-6 text-slate-400 shrink-0" />
                        <div className="space-y-1">
                            <h4 className="font-black text-slate-900 text-xs">Restricted Data</h4>
                            <p className="text-[10px] text-slate-500 font-bold leading-relaxed">Compliance Summary, Bias Analysis, and STR modules are {role === 'system_admin' ? 'currently optimizing' : 'hidden for your role'}. Contact a Senior Admin for elevated policy access.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
