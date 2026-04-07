'use client';

import {
    ShieldCheck,
    FileText,
    History as HistoryIcon,
    ArrowUpRight,
    Settings,
    Search,
    Download,
    BarChart3,
    CheckCircle2,
    SearchCheck,
    Lock,
    Scale,
    Building2,
    ChevronRight,
    ExternalLink
} from 'lucide-react';
import { useRole } from '@/app/context/RoleContext';
import Link from 'next/link';

export default function RegulatoryPortalPage() {
    const { role } = useRole();

    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="bg-blue-100 text-blue-700 text-[10px] font-black px-2 py-0.5 rounded-full uppercase tracking-widest border border-blue-200">
                            {role === 'system_admin' ? 'Strategic Compliance Oversight' : 'Compliance Reference View'}
                        </span>
                        <span className="text-slate-400 text-xs flex items-center gap-1 font-bold">
                            <Lock className="w-3 h-3" /> Bank of Ghana Directive v4.1
                        </span>
                    </div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Regulatory & Audit Portal</h1>
                    <p className="text-slate-500 text-sm mt-1 max-w-2xl font-medium leading-relaxed">
                        {role === 'system_admin' ? 'Strategic framework for institutional compliance monitoring, policy governance, and statutory filing.' :
                            'Read-only compliance guidance and statutory filing status for internal auditing.'}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button className="flex items-center gap-2 px-5 py-2.5 border border-slate-200 bg-white rounded-2xl text-xs font-black text-slate-700 shadow-sm hover:bg-slate-50 transition-all">
                        <Download className="w-4 h-4" /> Download Full Archive
                    </button>
                    <button className="flex items-center gap-2 px-5 py-2.5 bg-slate-900 text-white rounded-2xl text-xs font-black shadow-lg shadow-slate-200 hover:bg-slate-800 transition-all">
                        <Settings className="w-4 h-4" /> Policy Settings
                    </button>
                </div>
            </div>

            {/* Compliance KPI Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="bg-white p-8 rounded-[40px] border border-slate-200 shadow-sm flex flex-col justify-between group cursor-default">
                    <div className="flex justify-between items-start mb-6">
                        <div className="p-3 bg-blue-50 rounded-2xl text-blue-600 transition-transform group-hover:scale-110">
                            <Scale className="w-6 h-6" />
                        </div>
                        <span className="flex items-center gap-1 text-xs font-black text-emerald-600 bg-emerald-50 px-2 py-1 rounded-full uppercase">
                            <ArrowUpRight className="w-3 h-3" /> 100%
                        </span>
                    </div>
                    <div>
                        <h3 className="text-3xl font-black text-slate-900">Compliance Health</h3>
                        <p className="text-xs text-slate-400 font-bold uppercase tracking-widest mt-1">Regulatory Alignment Score</p>
                    </div>
                </div>

                <div className="bg-white p-8 rounded-[40px] border border-slate-200 shadow-sm flex flex-col justify-between group cursor-default">
                    <div className="flex justify-between items-start mb-6">
                        <div className="p-3 bg-emerald-50 rounded-2xl text-emerald-600 transition-transform group-hover:scale-110">
                            <FileText className="w-6 h-6" />
                        </div>
                        <span className="flex items-center gap-1 text-xs font-black text-blue-600 bg-blue-50 px-2 py-1 rounded-full uppercase">
                            84 / 92
                        </span>
                    </div>
                    <div>
                        <h3 className="text-3xl font-black text-slate-900">STR Submissions</h3>
                        <p className="text-xs text-slate-400 font-bold uppercase tracking-widest mt-1">Suspicious Transaction Reports</p>
                    </div>
                </div>

                <div className="bg-white p-8 rounded-[40px] border border-slate-200 shadow-sm flex flex-col justify-between group cursor-default">
                    <div className="flex justify-between items-start mb-6">
                        <div className="p-3 bg-indigo-50 rounded-2xl text-indigo-600 transition-transform group-hover:scale-110">
                            <HistoryIcon className="w-6 h-6" />
                        </div>
                        <span className="flex items-center gap-1 text-xs font-black text-orange-600 bg-orange-50 px-2 py-1 rounded-full uppercase">
                            12 Pending
                        </span>
                    </div>
                    <div>
                        <h3 className="text-3xl font-black text-slate-900">Internal Audits</h3>
                        <p className="text-xs text-slate-400 font-bold uppercase tracking-widest mt-1">Pending Operational Reviews</p>
                    </div>
                </div>
            </div>

            {/* Regional Risk Exposure */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Main Content (3/4) */}
                <div className="lg:col-span-3 space-y-6">
                    <div className="bg-white rounded-[40px] border border-slate-200 shadow-sm overflow-hidden p-8">
                        <div className="flex items-center justify-between mb-8">
                            <h2 className="text-xl font-black text-slate-900 tracking-tight flex items-center gap-2">
                                <Building2 className="w-6 h-6 text-blue-600" /> Statutory Filing Integrity
                            </h2>
                            <button className="text-xs font-black text-blue-600 hover:underline">VIEW ALL FILINGS</button>
                        </div>
                        <div className="space-y-6">
                            {[
                                { law: 'Anti-Money Laundering Act, 2020', code: 'Act 1044', status: 'In Sync', color: 'text-emerald-500 bg-emerald-100/30' },
                                { law: 'Payment Systems & Services Act', code: 'Act 987', status: 'In Sync', color: 'text-emerald-500 bg-emerald-100/30' },
                                { law: 'Data Protection Directive', code: 'BoG G4.1', status: 'Pending Review', color: 'text-orange-500 bg-orange-100/30 transition-all' },
                                { law: 'Cybersecurity Regulation', code: 'CS 2011', status: 'Delayed Sync', color: 'text-red-500 bg-red-100/30' }
                            ].map((item, i) => (
                                <div key={i} className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl border border-slate-100 hover:bg-white hover:border-blue-200 hover:shadow-lg hover:shadow-blue-50 transition-all group cursor-pointer">
                                    <div className="flex items-center gap-4">
                                        <div className="w-10 h-10 rounded-xl bg-white flex items-center justify-center text-slate-400 shadow-sm font-black text-xs">
                                            {item.code.split(' ')[1]}
                                        </div>
                                        <div>
                                            <p className="text-xs font-black text-slate-900 uppercase tracking-tight">{item.law}</p>
                                            <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">{item.code}</p>
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-6">
                                        <span className={`px-2 py-1 rounded-lg text-[10px] font-black uppercase tracking-widest border ${item.color.replace('text-', 'border-').replace('bg-', 'bg-opacity-20 border-').replace('100/30', '200')}`}>
                                            {item.status}
                                        </span>
                                        <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-blue-600 transition-colors" />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Recent STR Reports */}
                    <div className="bg-white rounded-[40px] border border-slate-200 shadow-sm overflow-hidden p-8">
                        <div className="flex items-center justify-between mb-8">
                            <h2 className="text-xl font-black text-slate-900 tracking-tight flex items-center gap-2">
                                <SearchCheck className="w-6 h-6 text-blue-600" /> High-Risk STR Queue
                            </h2>
                            <Link href="/dashboard/regulatory/lineage" className="text-xs font-black text-slate-400 hover:text-blue-600 transition-colors uppercase tracking-[0.2em]">VIEW LINEAGE VIEWER</Link>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left">
                                <thead className="text-[10px] font-black text-slate-400 uppercase tracking-widest border-b border-slate-100 font-mono">
                                    <tr>
                                        <th className="pb-4">CASE_ID</th>
                                        <th className="pb-4">FILING_TYPE</th>
                                        <th className="pb-4">INSTITUTIONAL_RISK</th>
                                        <th className="pb-4">STR_STATUS</th>
                                        <th className="pb-4 text-right">ACTION</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-50">
                                    {[
                                        { id: 'STR-GH-9921', type: 'SAR / AML', risk: 'HIGH', status: 'Submitted', color: 'text-emerald-500' },
                                        { id: 'STR-GH-8821', type: 'Structuring', risk: 'CRITICAL', status: 'Pending', color: 'text-orange-500' },
                                        { id: 'STR-GH-7743', type: 'UBO Conflict', risk: 'MEDIUM', status: 'In Draft', color: 'text-blue-500' },
                                    ].map((row, i) => (
                                        <tr key={i} className="hover:bg-slate-50 transition-colors group">
                                            <td className="py-5 font-mono font-black text-slate-900">{row.id}</td>
                                            <td className="py-5 text-[10px] font-black text-slate-400 uppercase tracking-widest">{row.type}</td>
                                            <td className="py-5">
                                                <span className={`text-[10px] font-black uppercase tracking-widest ${row.risk === 'CRITICAL' ? 'text-red-600' : row.risk === 'HIGH' ? 'text-orange-600' : 'text-blue-600'}`}>
                                                    {row.risk}
                                                </span>
                                            </td>
                                            <td className="py-5">
                                                <div className="flex items-center gap-2">
                                                    <div className={`w-2 h-2 rounded-full ${row.status === 'Submitted' ? 'bg-emerald-500' : row.status === 'Pending' ? 'bg-orange-500 animate-pulse' : 'bg-slate-400'}`} />
                                                    <span className="text-xs font-black text-slate-700">{row.status}</span>
                                                </div>
                                            </td>
                                            <td className="py-5 text-right">
                                                <button className="text-blue-600 font-bold hover:underline text-xs tracking-tighter uppercase whitespace-nowrap">Review Filing</button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Sidebar (1/4) */}
                <div className="space-y-6">
                    {/* Compliance Alert */}
                    <div className="bg-red-600 rounded-[40px] p-8 text-white shadow-2xl shadow-red-100 flex flex-col justify-between h-full min-h-[400px]">
                        <div>
                            <h3 className="text-2xl font-black tracking-tight mb-4 leading-none uppercase">Auditor Access Logged</h3>
                            <p className="text-sm font-medium text-red-100 opacity-90 leading-relaxed">
                                External audit firm (PWC) has requested source access to bias data. Authorization currently on legal hold.
                            </p>
                        </div>
                        <div className="space-y-4">
                            <div className="p-4 bg-white/10 rounded-3xl border border-white/20">
                                <p className="text-[10px] font-black text-red-200 uppercase tracking-widest mb-1">Status</p>
                                <p className="font-bold text-lg">LEGAL HOLD ACT-112</p>
                            </div>
                            {role === 'system_admin' ? (
                                <button className="w-full py-4 bg-white text-red-600 rounded-2xl font-black text-sm shadow-xl hover:bg-slate-50 transition-all">Resolve Authorization</button>
                            ) : (
                                <div className="p-4 bg-red-900/40 rounded-2xl border border-red-400/20 text-center">
                                    <p className="text-[10px] font-black text-red-200 uppercase tracking-widest mb-1">Analyst Restriction</p>
                                    <p className="text-[9px] font-bold text-red-100">Elevate to System Admin for resolution</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Report Export Box */}
                    <div className="bg-white rounded-[40px] border border-slate-200 p-8 shadow-sm">
                        <h3 className="font-black text-slate-900 text-sm uppercase tracking-[0.2em] mb-6">Internal Controls</h3>
                        <div className="space-y-4">
                            <button className="w-full flex items-center justify-between p-4 bg-slate-50 rounded-2xl border border-slate-100 hover:bg-white hover:border-blue-200 transition-all group">
                                <div className="flex items-center gap-3">
                                    <FileText className="w-4 h-4 text-slate-400" />
                                    <span className="text-xs font-black text-slate-700 uppercase tracking-tight">Q4 SAR Report</span>
                                </div>
                                <Download className="w-4 h-4 text-slate-300 group-hover:text-blue-600" />
                            </button>
                            <button className="w-full flex items-center justify-between p-4 bg-slate-50 rounded-2xl border border-slate-100 hover:bg-white hover:border-blue-200 transition-all group">
                                <div className="flex items-center gap-3">
                                    <HistoryIcon className="w-4 h-4 text-slate-400" />
                                    <span className="text-xs font-black text-slate-700 uppercase tracking-tight">Audit Trail v2</span>
                                </div>
                                <Download className="w-4 h-4 text-slate-300 group-hover:text-blue-600" />
                            </button>
                            <button className="w-full mt-4 py-4 border-2 border-slate-100 text-slate-500 rounded-2xl text-[10px] font-black uppercase tracking-widest hover:border-blue-200 hover:text-blue-600 transition-all">Configure Automations</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
