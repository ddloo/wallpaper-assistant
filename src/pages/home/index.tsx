import { Link } from 'react-router-dom'
import styles from './index.module.scss'

function Home() {
  return (
    <div className={styles.container}>
      <div className={styles.content}>
        <p className="mb-6">欢迎使用壁纸助手，这是一个帮助您管理和自动切换桌面壁纸的应用程序。</p>
        
        <div className={styles.actions}>
          <button className={styles.actionButton}>选择壁纸</button>
          <Link to="/settings" className={styles.navLink}>前往设置</Link>
        </div>
      </div>
    </div>
  )
}

export default Home
