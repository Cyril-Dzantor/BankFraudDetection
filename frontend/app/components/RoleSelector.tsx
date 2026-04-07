'use client';

import { useRole, UserRole } from '@/app/context/RoleContext';
import {
    ChevronDown,
    ChevronUp,
    Shield,
    ShieldAlert,
    ShieldCheck,
    ShieldPlus,
    Eye,
} from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';

export default function RoleSelector({ expanded = false }: { expanded?: boolean }) {
    const { role, setRole, userName } = useRole();
    const [isOpen, setIsOpen] = useState(false);

    const roles: { id: UserRole; label: string; icon: any; color: string; bg: string; border: string; desc: string; title: string }[] = [
        {
            id: 'junior_analyst',
            label: 'Junior Analyst',
            icon: Shield,
            color: 'text-blue-600',
            bg: 'bg-blue-50',
            border: 'border-blue-100',
            desc: 'Read-only monitoring',
            title: 'Jr. Analyst'
        },
        {
            id: 'senior_analyst',
            label: 'Senior Analyst',
            icon: ShieldAlert,
            color: 'text-red-600',
            bg: 'bg-red-50',
            border: 'border-red-100',
            desc: 'Full case resolution',
            title: 'Sr. Analyst'
        },
        {
            id: 'system_admin',
            label: 'System Admin',
            icon: ShieldCheck,
            color: 'text-indigo-600',
            bg: 'bg-indigo-50',
            border: 'border-indigo-100',
            desc: 'IAM & System Oversight',
            title: 'Sys Admin'
        },
        {
            id: 'compliance_lead',
            label: 'Compliance Lead',
            icon: ShieldPlus,
            color: 'text-emerald-600',
            bg: 'bg-emerald-50',
            border: 'border-emerald-100',
            desc: 'Maker-Checker Oversight',
            title: 'Compliance Lead'
        }
    ];

    const currentRole = roles.find(r => r.id === role) || roles[0];

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-2xl hover:bg-slate-50 transition-all group border border-transparent hover:border-slate-200 ${expanded ? 'aspect-auto' : ''}`}
            >
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white font-bold overflow-hidden shadow-sm group-hover:shadow-md transition-shadow">
                        <Image src={`https://api.dicebear.com/9.x/avataaars/svg?seed=${userName || 'Kwame'}&backgroundColor=2563eb`} alt="avatar" width={40} height={40} className="w-full h-full object-cover" />
                    </div>
                    <div className="text-left">
                        <p className="text-sm font-bold text-slate-900 leading-tight">{userName || 'Kwame Mensah'}</p>
                        <p className="text-xs text-slate-500 font-medium">{currentRole.title}</p>
                    </div>
                </div>
                <ChevronUp className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="absolute bottom-full left-0 right-0 mb-2 p-2 bg-white border border-slate-200 rounded-3xl shadow-2xl z-50 animate-in fade-in slide-in-from-bottom-2 duration-200">
                    <div className="px-3 py-2 mb-2 border-b border-slate-100">
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
                            <Eye className="w-3 h-3" /> Switch Perspective
                        </span>
                    </div>
                    {roles.map((r) => (
                        <button
                            key={r.id}
                            onClick={() => {
                                setRole(r.id);
                                setIsOpen(false);
                            }}
                            className={`w-full flex items-center gap-3 p-3 rounded-2xl transition-all mb-1 last:mb-0
                                ${role === r.id ? 'bg-blue-600 text-white shadow-lg shadow-blue-200' : 'hover:bg-slate-50 text-slate-600'}
                            `}
                        >
                            <div className={`p-2 rounded-xl ${role === r.id ? 'bg-white/20' : r.color}`}>
                                <r.icon className="w-4 h-4" />
                            </div>
                            <div className="text-left">
                                <p className={`text-xs font-black leading-none mb-0.5 ${role === r.id ? 'text-white' : 'text-slate-900'}`}>
                                    {r.label}
                                </p>
                                <p className={`text-[9px] font-bold ${role === r.id ? 'text-blue-100' : 'text-slate-400'}`}>
                                    {r.desc}
                                </p>
                            </div>
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}
