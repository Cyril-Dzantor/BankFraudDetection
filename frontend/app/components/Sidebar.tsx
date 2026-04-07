'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
    LayoutDashboard,
    BellRing,
    Users,
    BarChart3,
    FileText,
    Settings,
    ShieldAlert,
    Network,
    Briefcase,
    BrainCircuit,
    History as HistoryIcon,
    ShieldCheck,
    User
} from 'lucide-react';

import RoleSelector from './RoleSelector';

export default function Sidebar() {
    const pathname = usePathname();

    const isActive = (href: string) => {
        if (href === '/dashboard') {
            return pathname === '/dashboard';
        }
        return pathname.startsWith(href);
    };

    const navItems = [
        { href: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
        { href: '/dashboard/alerts', icon: BellRing, label: 'Alert Queue', badge: '12' },
        { href: '/dashboard/cases', icon: Briefcase, label: 'Case Management' },
        // { href: '/network', icon: Network, label: 'Network Explorer' },
        { href: '/dashboard/models', icon: BrainCircuit, label: 'Risk Models' },
        { href: '/dashboard/accounts', icon: Users, label: 'Customer Directory' },
        { href: '/dashboard/profile', icon: User, label: 'My Account' },
    ];

    const reportItems = [
        { href: '/dashboard', icon: BarChart3, label: 'Analytics' },
        // { href: '/dashboard/bias-fairness', icon: ShieldAlert, label: 'Bias & Fairness' },
        // { href: '/dashboard/regulatory/lineage', icon: HistoryIcon, label: 'Decision Lineage' },
        { href: '/dashboard/audit', icon: FileText, label: 'Audit Logs' },
        { href: '/dashboard/admin/provisioning', icon: Settings, label: 'Admin Provisioning' },
    ];

    return (
        <aside className="w-72 bg-white border-r border-slate-200 flex flex-col h-full shadow-sm z-50">
            {/* Logo */}
            <div className="p-8 pb-6">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
                        <ShieldCheck className="text-white w-6 h-6" />
                    </div>
                    <span className="text-xl font-black text-slate-900 tracking-tight">FraudSense <span className="text-blue-600">AI</span></span>
                </div>
            </div>


            <div className="flex-1 overflow-y-auto py-6 flex flex-col gap-6 px-4">
                <nav className="space-y-1">
                    {navItems.map((item) => {
                        const active = isActive(item.href);
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={`flex items-center justify-between px-3 py-2.5 rounded-lg font-medium text-sm transition-colors ${active
                                    ? 'bg-blue-50 text-blue-700'
                                    : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                                    }`}
                            >
                                <div className="flex items-center gap-3">
                                    <item.icon className="w-5 h-5" />
                                    {item.label}
                                </div>
                                {item.badge && (
                                    <span className="bg-red-100 text-red-600 py-0.5 px-2 rounded-full text-xs font-bold">
                                        {item.badge}
                                    </span>
                                )}
                            </Link>
                        );
                    })}
                </nav>

                <div>
                    <h3 className="px-3 text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Reports</h3>
                    <nav className="space-y-1">
                        {reportItems.map((item) => {
                            const active = isActive(item.href);
                            return (
                                <Link
                                    key={item.label}
                                    href={item.href}
                                    className={`flex items-center gap-3 px-3 py-2.5 rounded-lg font-medium text-sm transition-colors ${active
                                        ? 'bg-blue-50 text-blue-700'
                                        : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                                        }`}
                                >
                                    <item.icon className="w-5 h-5" />
                                    {item.label}
                                </Link>
                            );
                        })}
                    </nav>
                </div>
            </div>

            <div className="p-4 border-t border-slate-200">
                <Link
                    href="#"
                    className={`flex items-center gap-3 px-3 py-2.5 mb-2 rounded-lg font-medium text-sm transition-colors ${isActive('#')
                        ? 'bg-blue-50 text-blue-700'
                        : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                        }`}
                >
                    <Settings className="w-5 h-5" />
                    Settings
                </Link>

                {/* Integrated Role Selector & Profile */}
                <RoleSelector expanded={true} />
            </div>
        </aside>
    );
}
