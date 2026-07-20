import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';

const navClass = ({ isActive }: { isActive: boolean }) =>
  isActive ? 'nav-link nav-link--active' : 'nav-link';

/** App chrome: product header + primary nav, with page content rendered as
 * children. Shared by every page. */
export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__brand">Kompress</div>
        <nav className="app-nav">
          <NavLink to="/" end className={navClass}>
            Dashboard
          </NavLink>
          <NavLink to="/submit" className={navClass}>
            Submit
          </NavLink>
          <NavLink to="/deployments" className={navClass}>
            Deployments
          </NavLink>
        </nav>
      </header>
      <main className="app-main">{children}</main>
    </div>
  );
}
