import Sidebar from '@/app/components/Sidebar';
import { RoleProvider } from '@/app/context/RoleContext';

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <RoleProvider>
            <div className="flex h-screen bg-slate-50 overflow-hidden font-sans selection:bg-blue-100 selection:text-blue-900">
                <Sidebar />
                <main className="flex-1 overflow-y-auto px-8 pt-8 pb-12 custom-scrollbar">
                    {children}
                </main>
            </div>
        </RoleProvider>
    );
}
