'use client';

import {
    Download,
    Filter,
    Search,
    ShieldCheck,
    UserX,
    Database,
    Lock,
    Calendar,
    ChevronDown,
    AlertCircle
} from 'lucide-react';
import { toast } from 'sonner';

import { useState, useEffect, useMemo } from 'react';
import { useRole } from '@/app/context/RoleContext';
import { getBackendUrl } from '@/app/utils/api';

export default function AuditPortalPage() {
    const { role } = useRole();
    const [auditLogs, setAuditLogs] = useState<any[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const backendUrl = getBackendUrl();

    useEffect(() => {
        const fetchLogs = async () => {
            if (role !== 'system_admin' && role !== 'compliance_lead') {
                setIsLoading(false);
                return;
            }
            try {
                const res = await fetch(`${backendUrl}/api/v1/audit/`, {
                    headers: { 'X-User-Role': role }
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data.items) {
                        setAuditLogs(data.items);
                    }
                }
            } catch (error) {
                console.error('Failed to fetch audit logs', error);
            } finally {
                setIsLoading(false);
            }
        };
        fetchLogs();
    }, [backendUrl, role]);
    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">

            {/* Header Bar */}
            <div className="flex items-start justify-between">
                <div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Regulatory & Audit Portal</h1>
                    <p className="text-slate-500 text-sm mt-1">Immutable system logs, access records, and compliance monitoring.</p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => toast('Date range picker opened')}
                        className="flex items-center gap-2 px-4 py-2 border border-slate-200 bg-white rounded-lg text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-colors"
                    >
                        <Calendar className="w-4 h-4" /> Mar 1 - Mar 8 <ChevronDown className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => window.open(`${backendUrl}/api/v1/audit/export`, '_blank')}
                        className="flex items-center gap-2 px-4 py-2 bg-slate-900 border border-transparent rounded-lg text-sm font-semibold text-white shadow-sm hover:bg-slate-800 transition-colors"
                    >
                        <Download className="w-4 h-4" /> Export Compliance Pack
                    </button>
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {/* Compliance Score */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">Compliance Score</span>
                        <ShieldCheck className="w-5 h-5 text-emerald-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">99.8%</h3>
                        <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded">ISO 27001 Status</span>
                    </div>
                </div>

                {/* Failed Access Attemps */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">Failed Access</span>
                        <UserX className="w-5 h-5 text-red-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">42</h3>
                        <span className="text-xs font-bold text-slate-500">Last 24h</span>
                    </div>
                </div>

                {/* Data Exports */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">Data Exports</span>
                        <Database className="w-5 h-5 text-blue-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">12</h3>
                        <span className="text-xs font-bold text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded">3 require review</span>
                    </div>
                </div>

                {/* Privileged Actions */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                    <div className="flex justify-between items-start mb-2">
                        <span className="text-slate-500 text-sm font-medium">Privileged Actions</span>
                        <Lock className="w-5 h-5 text-purple-500" />
                    </div>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-3xl font-bold text-slate-900 leading-none">8</h3>
                        <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded">All Verified</span>
                    </div>
                </div>
            </div>

            {/* Filters and Search */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-col md:flex-row gap-4 items-center justify-between">
                <div className="flex flex-1 w-full max-w-md relative">
                    <Search className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                    <input
                        type="text"
                        placeholder="Search by ID, actor, or IP address..."
                        className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-slate-900 transition-shadow"
                    />
                </div>

                <div className="flex items-center gap-3 w-full md:w-auto overflow-x-auto pb-1 md:pb-0">
                    <button className="flex items-center gap-2 px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors whitespace-nowrap">
                        Action: All <ChevronDown className="w-4 h-4" />
                    </button>
                    <button className="flex items-center gap-2 px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors whitespace-nowrap">
                        Status: Any <ChevronDown className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => toast('Advanced filter panel opened')}
                        className="flex items-center gap-2 px-3 py-2 bg-slate-100 border border-slate-200 rounded-lg text-sm font-bold text-slate-700 hover:bg-slate-200 transition-colors whitespace-nowrap"
                    >
                        <Filter className="w-4 h-4" /> Advanced
                    </button>
                </div>
            </div>

            {/* Audit Data Table */}
            {role !== 'system_admin' && role !== 'compliance_lead' ? (
                <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-20 flex flex-col items-center justify-center text-center">
                    <div className="w-16 h-16 bg-red-50 rounded-2xl flex items-center justify-center mb-4">
                        <Lock className="w-8 h-8 text-red-300" />
                    </div>
                    <h3 className="text-lg font-black text-slate-900">Access Restricted</h3>
                    <p className="text-sm text-slate-500 font-medium max-w-xs mt-1">
                        Viewing immutable system audit logs requires **System Admin** or **Compliance Lead** authorization.
                    </p>
                </div>
            ) : (
                <div className="bg-white border border-slate-200 rounded-xl shadow-sm overflow-hidden">
                    {/* ... table content remains same as before but wrapped ... */}
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-left">
                            <thead className="text-xs text-slate-500 font-bold uppercase tracking-wider bg-slate-50 border-b border-slate-200">
                                <tr>
                                    <th className="px-6 py-4">Timestamp / ID</th>
                                    <th className="px-6 py-4">Actor & IP</th>
                                    <th className="px-6 py-4">Action</th>
                                    <th className="px-6 py-4">Resource</th>
                                    <th className="px-6 py-4">Status & Risk</th>
                                    <th className="px-6 py-4 text-right">Details</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100 font-mono text-sm">
                                {isLoading ? (
                                    <tr>
                                        <td colSpan={6} className="px-6 py-12 text-center text-slate-400 font-sans italic">
                                            Decrypting logs...
                                        </td>
                                    </tr>
                                ) : auditLogs.length === 0 ? (
                                    <tr>
                                        <td colSpan={6} className="px-6 py-12 text-center text-slate-400 font-sans italic">
                                            No logs found for the selected period.
                                        </td>
                                    </tr>
                                ) : (
                                    auditLogs.map((log) => (
                                        <tr key={log.id} className="hover:bg-slate-50 transition-colors group">
                                            {/* ... existing cells ... */}
                                            <td className="px-6 py-4">
                                                <p className="font-semibold text-slate-900">{log.timestamp}</p>
                                                <p className="text-xs text-slate-400 mt-0.5">{log.id}</p>
                                            </td>

                                            <td className="px-6 py-4">
                                                <p className="font-semibold text-slate-700">{log.actor}</p>
                                                <p className="text-xs text-slate-400 mt-0.5">{log.ip}</p>
                                            </td>

                                            <td className="px-6 py-4">
                                                <span className="px-2 py-1 bg-slate-100 text-slate-700 rounded text-xs font-bold tracking-tight">
                                                    {log.action}
                                                </span>
                                            </td>

                                            <td className="px-6 py-4 text-slate-600">
                                                {log.resource}
                                            </td>

                                            <td className="px-6 py-4">
                                                <div className="flex flex-col gap-1.5 items-start">
                                                    <span className={`flex items-center gap-1.5 text-xs font-bold
                                ${log.status === 'Success' ? 'text-emerald-600' :
                                                            log.status === 'Failed' ? 'text-red-600' :
                                                                'text-amber-600'}`}>
                                                        {log.status === 'Success' ? <ShieldCheck className="w-3.5 h-3.5" /> : <AlertCircle className="w-3.5 h-3.5" />}
                                                        {log.status}
                                                    </span>
                                                    <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider
                                ${log.risk === 'Critical' ? 'bg-red-100 text-red-700' :
                                                            log.risk === 'High' ? 'bg-orange-100 text-orange-700' :
                                                                log.risk === 'Medium' ? 'bg-amber-100 text-amber-700' :
                                                                    'bg-slate-100 text-slate-600'}`}>
                                                        {log.risk} Risk
                                                    </span>
                                                </div>
                                            </td>

                                            <td className="px-6 py-4 text-right">
                                                <button
                                                    onClick={() => {
                                                        navigator.clipboard.writeText(log.payload || '{"message": "No additional payload context"}');
                                                        toast.success('Payload JSON copied to clipboard!');
                                                    }}
                                                    className="px-3 py-1.5 border border-slate-200 rounded-lg text-xs font-bold text-slate-600 hover:bg-slate-100 hover:border-slate-300 transition-all shadow-sm font-sans block ml-auto"
                                                >
                                                    View JSON
                                                </button>
                                            </td>
                                        </tr>
                                    ))
                                )}
                            </tbody>
                        </table>
                    </div>
                    {/* ... details pagination ... */}
                    <div className="px-6 py-4 border-t border-slate-200 bg-slate-50 text-xs font-sans text-slate-500 flex justify-between items-center">
                        <span>Showing {auditLogs.length} logs</span>
                        <div className="flex gap-1">
                            <button className="px-3 py-1 border border-slate-200 rounded font-medium hover:bg-white transition-colors">Prev</button>
                            <button className="px-3 py-1 bg-slate-900 text-white border border-slate-900 rounded font-medium shadow-sm">1</button>
                            <button className="px-3 py-1 border border-slate-200 rounded font-medium hover:bg-white transition-colors">Next</button>
                        </div>
                    </div>
                </div>
            )}

        </div>
    );
}
