'use client';

import {
    ShieldCheck,
    AlertTriangle,
    Users,
    Activity,
    Map as MapIcon,
    ArrowUpRight,
    ArrowDownRight,
    Filter,
    Download
} from 'lucide-react';

export default function BiasFairnessPage() {
    return (
        <div className="max-w-7xl mx-auto space-y-6 pb-12">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <div className="flex items-center gap-2 mb-1">
                        <span className="bg-emerald-100 text-emerald-700 text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider">Audit Active</span>
                        <span className="text-slate-400 text-xs flex items-center gap-1">
                            <Activity className="w-3 h-3" /> Last updated: 14 mins ago
                        </span>
                    </div>
                    <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">Bias & Fairness Audit</h1>
                    <p className="text-slate-500 text-sm mt-1 max-w-2xl">
                        Monitoring algorithmic fairness across demographics for Bank of Ghana compliance.
                        Analysis includes regional distribution, gender parity, and age group impact assessments.
                    </p>
                </div>
                <div className="flex items-center gap-2">
                    <button className="flex items-center gap-2 px-4 py-2 border border-slate-200 bg-white rounded-lg text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-colors">
                        <Download className="w-4 h-4" /> Export Data
                    </button>
                    <button className="flex items-center gap-2 px-4 py-2 border border-slate-200 bg-white rounded-lg text-sm font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-colors">
                        <Filter className="w-4 h-4" /> Filters
                    </button>
                </div>
            </div>

            {/* Critical Warning */}
            <div className="bg-orange-50 border border-orange-200 rounded-xl p-4 flex items-start gap-4 shadow-sm">
                <div className="w-10 h-10 rounded-lg bg-orange-100 flex items-center justify-center text-orange-600 shrink-0">
                    <AlertTriangle className="w-6 h-6" />
                </div>
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-bold text-orange-900">Fairness Deviation Warning</h3>
                        <span className="bg-orange-200 text-orange-800 text-[10px] font-black px-1.5 py-0.5 rounded uppercase">Critical</span>
                    </div>
                    <p className="text-sm text-orange-800 leading-relaxed">
                        Block rates for users in the <span className="font-bold">Northern Region</span> exceed the 5% deviation threshold compared to the national average.
                        Immediate investigation recommended for compliance.
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button className="text-orange-900 text-sm font-bold hover:underline">Dismiss</button>
                    <button className="bg-orange-600 text-white px-4 py-2 rounded-lg text-sm font-bold shadow-sm hover:bg-orange-700 transition-colors">
                        View Details
                    </button>
                </div>
            </div>

            {/* KPI Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-slate-800">
                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                    <div className="flex justify-between items-start mb-4">
                        <span className="text-slate-500 text-sm font-medium">Audited Decisions</span>
                        <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                            <Activity className="w-5 h-5" />
                        </div>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-3xl font-black">1.24M</h3>
                        <span className="flex items-center gap-0.5 text-xs font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded">
                            <ArrowUpRight className="w-3 h-3" /> +12%
                        </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">vs last month</p>
                </div>

                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                    <div className="flex justify-between items-start mb-4">
                        <span className="text-slate-500 text-sm font-medium">Avg. Approval Rate</span>
                        <div className="p-2 bg-indigo-50 rounded-lg text-indigo-600">
                            <ShieldCheck className="w-5 h-5" />
                        </div>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-3xl font-black">88.4%</h3>
                        <span className="flex items-center gap-0.5 text-xs font-bold text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded">
                            stable metric
                        </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">across all demographics</p>
                </div>

                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                    <div className="flex justify-between items-start mb-4">
                        <span className="text-slate-500 text-sm font-medium">Fairness Score</span>
                        <div className="p-2 bg-purple-50 rounded-lg text-purple-600">
                            <Users className="w-5 h-5" />
                        </div>
                    </div>
                    <div>
                        <div className="flex items-baseline gap-2">
                            <h3 className="text-3xl font-black text-blue-600">98.2</h3>
                            <span className="text-slate-400 font-bold text-base">/ 100</span>
                        </div>
                        <div className="w-full bg-slate-100 h-2 rounded-full mt-3 overflow-hidden">
                            <div className="bg-blue-600 h-full w-[98%]" />
                        </div>
                    </div>
                </div>

                <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
                    <div className="flex justify-between items-start mb-4">
                        <span className="text-slate-500 text-sm font-medium">Pending Reviews</span>
                        <div className="p-2 bg-orange-50 rounded-lg text-orange-600">
                            <AlertTriangle className="w-5 h-5" />
                        </div>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <h3 className="text-3xl font-black">45</h3>
                        <span className="flex items-center gap-0.5 text-xs font-bold text-red-600 bg-red-50 px-1.5 py-0.5 rounded">
                            <ArrowDownRight className="w-3 h-3" /> -5%
                        </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">improving response time</p>
                </div>
            </div>

            {/* Analysis Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Ghana Map Analysis */}
                <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
                    <div className="p-6 border-b border-slate-100 flex items-center justify-between">
                        <div className="flex items-center gap-3 text-slate-800">
                            <div className="p-2 bg-blue-50 rounded-lg text-blue-600">
                                <MapIcon className="w-5 h-5" />
                            </div>
                            <h2 className="font-bold text-lg">Regional Impact Analysis</h2>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-slate-500 bg-slate-100 px-3 py-1.5 rounded-lg border border-slate-200">
                                Block Rate Deviation
                            </span>
                        </div>
                    </div>
                    <div className="flex-1 p-6 bg-slate-50 relative min-h-[400px] flex items-center justify-center">
                        {/* Placeholder for Map visualization */}
                        <div className="absolute inset-0 grayscale opacity-20 pointer-events-none">
                            <div className="w-full h-full bg-[radial-gradient(circle_at_50%_50%,#e2e8f0_1px,transparent_1px)] bg-size-[24px_24px]" />
                        </div>
                        <div className="relative w-full h-full flex items-center justify-center">
                            <div className="relative w-72 h-96 bg-slate-200 rounded-[20%] rotate-10 blur-3xl opacity-30" />
                            {/* Hotspots */}
                            <div className="absolute top-1/4 left-1/3 w-4 h-4 bg-emerald-500 rounded-full border-4 border-white shadow-lg animate-pulse" />
                            <div className="absolute bottom-1/3 right-1/4 w-6 h-6 bg-orange-500 rounded-full border-4 border-white shadow-lg animate-pulse" />
                            <div className="absolute bottom-1/4 left-1/2 w-4 h-4 bg-emerald-500 rounded-full border-4 border-white shadow-lg animate-pulse" />

                            <div className="absolute top-12 left-12 bg-white/90 backdrop-blur p-3 rounded-lg border border-slate-200 shadow-xl max-w-[200px]">
                                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">GHANA REGIONAL HEATMAP</p>
                                <p className="text-xs text-slate-600 leading-relaxed">Interactive node distribution focusing on approval variance per region.</p>
                            </div>
                        </div>

                        {/* Legend */}
                        <div className="absolute bottom-6 left-6 right-6 bg-white/80 backdrop-blur py-3 px-4 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between pointer-events-none">
                            <div className="flex items-center gap-6">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-emerald-500" />
                                    <span className="text-xs font-bold text-slate-600">Within Threshold (&lt;2%)</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-amber-500" />
                                    <span className="text-xs font-bold text-slate-600">Warning Zone (2-5%)</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full bg-orange-600" />
                                    <span className="text-xs font-bold text-slate-600">Non-Compliant (&gt;5%)</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Demographics */}
                <div className="space-y-6">
                    {/* Gender Parity */}
                    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
                        <div className="flex items-center gap-3 text-slate-800 mb-6">
                            <div className="p-2 bg-indigo-50 rounded-lg text-indigo-600">
                                <Users className="w-5 h-5" />
                            </div>
                            <h2 className="font-bold text-lg">Demographics</h2>
                        </div>

                        <div className="space-y-6">
                            <div>
                                <div className="flex justify-between items-end mb-2">
                                    <span className="text-xs font-black text-slate-500 uppercase tracking-wider">GENDER PARITY</span>
                                </div>
                                <div className="space-y-4">
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between items-center text-xs font-bold">
                                            <span className="text-slate-600">Male Approval Rate</span>
                                            <span className="text-slate-900 font-black">89.1%</span>
                                        </div>
                                        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                                            <div className="bg-blue-600 h-full w-[89.1%]" />
                                        </div>
                                    </div>
                                    <div className="space-y-1.5">
                                        <div className="flex justify-between items-center text-xs font-bold">
                                            <span className="text-slate-600">Female Approval Rate</span>
                                            <span className="text-slate-900 font-black">88.9%</span>
                                        </div>
                                        <div className="w-full h-2 bg-slate-100 rounded-full overflow-hidden">
                                            <div className="bg-blue-400 h-full w-[88.9%]" />
                                        </div>
                                    </div>
                                </div>
                                <div className="mt-4 p-3 bg-emerald-50 rounded-xl border border-emerald-100 flex items-center gap-3">
                                    <ShieldCheck className="w-5 h-5 text-emerald-600" />
                                    <p className="text-xs font-bold text-emerald-800">
                                        Parity difference is <span className="underline">0.2%</span> (Excellent)
                                    </p>
                                </div>
                            </div>

                            <hr className="border-slate-100" />

                            {/* Age Group */}
                            <div>
                                <div className="flex justify-between items-end mb-4">
                                    <span className="text-xs font-black text-slate-500 uppercase tracking-wider">AGE GROUP BLOCK RATE</span>
                                </div>
                                <div className="space-y-3">
                                    {[
                                        { label: '18-25', value: 12, color: 'bg-orange-500' },
                                        { label: '26-40', value: 4, color: 'bg-blue-600' },
                                        { label: '41-60', value: 2.5, color: 'bg-blue-600' },
                                        { label: '60+', value: 5, color: 'bg-blue-600' }
                                    ].map((age) => (
                                        <div key={age.label} className="flex items-center gap-3">
                                            <span className="w-10 text-[10px] font-black text-slate-500">{age.label}</span>
                                            <div className="flex-1 h-5 bg-slate-50 rounded-md overflow-hidden relative">
                                                <div className={`h-full ${age.color} rounded-r-sm`} style={{ width: `${age.value * 5}%` }} />
                                                <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] font-bold text-slate-900">{age.value}%</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                                <div className="mt-4 p-3 bg-red-50 rounded-xl border border-red-100 flex items-start gap-3">
                                    <AlertTriangle className="w-5 h-5 text-red-600 shrink-0" />
                                    <p className="text-xs font-bold text-red-800 leading-tight">
                                        High block rate for 18-25 group requires review of student loan criteria.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Fairness Alerts Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-6 border-b border-slate-100 flex items-center justify-between">
                    <h2 className="font-bold text-lg text-slate-800">Recent Fairness Alerts</h2>
                    <button className="text-blue-600 text-sm font-bold hover:underline">View All Logs</button>
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-slate-50 border-b border-slate-100">
                            <tr className="text-[10px] font-black text-slate-500 uppercase tracking-widest font-mono">
                                <th className="px-6 py-4">Alert ID</th>
                                <th className="px-6 py-4">Metric</th>
                                <th className="px-6 py-4">Segment / Context</th>
                                <th className="px-6 py-4">Severity</th>
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4 text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {[
                                { id: '#AL-9921', metric: 'Block Rate Deviation', segment: 'Region: Northern', severity: 'High', status: 'Investigating' },
                                { id: '#AL-9920', metric: 'Age Bias', segment: 'Age: 18-25', severity: 'Medium', status: 'Pending' },
                                { id: '#AL-9918', metric: 'Gender Gap', segment: 'Gender: Female', severity: 'Low', status: 'Resolved' },
                            ].map((alert) => (
                                <tr key={alert.id} className="hover:bg-slate-50 transition-colors group">
                                    <td className="px-6 py-4 font-mono font-bold text-slate-400 group-hover:text-slate-900 transition-colors">{alert.id}</td>
                                    <td className="px-6 py-4 font-bold text-slate-700">{alert.metric}</td>
                                    <td className="px-6 py-4 font-medium text-slate-500">{alert.segment}</td>
                                    <td className="px-6 py-4">
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-black uppercase tracking-wider
                                            ${alert.severity === 'High' ? 'bg-red-100 text-red-600' :
                                                alert.severity === 'Medium' ? 'bg-orange-100 text-orange-600' : 'bg-slate-100 text-slate-500'}`}>
                                            {alert.severity}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center gap-2">
                                            <div className={`w-2 h-2 rounded-full 
                                                ${alert.status === 'Resolved' ? 'bg-emerald-500' :
                                                    alert.status === 'Investigating' ? 'bg-amber-500 animate-pulse' : 'bg-slate-400'}`} />
                                            <span className="font-bold text-slate-600">{alert.status}</span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <button className="text-blue-600 font-bold hover:underline">
                                            {alert.status === 'Resolved' ? 'Details' : 'Review'}
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
