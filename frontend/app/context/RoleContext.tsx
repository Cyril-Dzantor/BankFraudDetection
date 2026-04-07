'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

export type UserRole = 'junior_analyst' | 'senior_analyst' | 'system_admin' | 'compliance_lead';

interface RoleContextType {
    role: UserRole;
    setRole: (role: UserRole) => void;
    userEmail: string | null;
    setUserEmail: (email: string | null) => void;
    userName: string | null;
    setUserName: (name: string | null) => void;
}

const RoleContext = createContext<RoleContextType | undefined>(undefined);

export function RoleProvider({ children }: { children: React.ReactNode }) {
    const [role, setRoleState] = useState<UserRole>('junior_analyst');
    const [userEmail, setUserEmailState] = useState<string | null>(null);
    const [userName, setUserNameState] = useState<string | null>(null);

    // Persist role in localStorage for demo stability
    useEffect(() => {
        const savedRole = localStorage.getItem('user-role') as UserRole;
        if (savedRole && ['junior_analyst', 'senior_analyst', 'system_admin', 'compliance_lead'].includes(savedRole)) {
            setRoleState(savedRole);
        }
        const savedEmail = localStorage.getItem('user-email');
        if (savedEmail) {
            setUserEmailState(savedEmail);
        }
        const savedName = localStorage.getItem('user-name');
        if (savedName) {
            setUserNameState(savedName);
        }
    }, []);

    const setRole = (newRole: UserRole) => {
        setRoleState(newRole);
        localStorage.setItem('user-role', newRole);
    };

    const setUserEmail = (email: string | null) => {
        setUserEmailState(email);
        if (email) {
            localStorage.setItem('user-email', email);
        } else {
            localStorage.removeItem('user-email');
        }
    };

    const setUserName = (name: string | null) => {
        setUserNameState(name);
        if (name) {
            localStorage.setItem('user-name', name);
        } else {
            localStorage.removeItem('user-name');
        }
    };

    return (
        <RoleContext.Provider value={{ role, setRole, userEmail, setUserEmail, userName, setUserName }}>
            {children}
        </RoleContext.Provider>
    );
}

export function useRole() {
    const context = useContext(RoleContext);
    if (context === undefined) {
        throw new Error('useRole must be used within a RoleProvider');
    }
    return context;
}
