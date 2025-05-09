import { Outlet } from 'react-router-dom'
import styles from './MainLayout.module.scss'

const MainLayout = () => {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>壁纸助手</h1>
      </header>
      
      <main>
        <Outlet />
      </main>
      
      <footer className={styles.footer}>
        <p>&copy; 2025 壁纸助手</p>
      </footer>
    </div>
  )
}

export default MainLayout
