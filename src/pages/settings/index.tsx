import { Link } from 'react-router-dom'
import styles from './index.module.scss'

function Settings() {
  return (
    <div className={styles.container}>
      <h2>设置</h2>
      <div className={styles.settingsForm}>
        <div className={styles.formGroup}>
          <label>壁纸源</label>
          <select>
            <option value="local">本地文件</option>
            <option value="online">在线服务</option>
          </select>
        </div>
        
        <div className={styles.formGroup}>
          <label>切换间隔</label>
          <select>
            <option value="300">5分钟</option>
            <option value="900">15分钟</option>
            <option value="1800">30分钟</option>
            <option value="3600">1小时</option>
          </select>
        </div>
        
        <div className={styles.buttonGroup}>
          <button className={styles.saveButton}>保存设置</button>
          <Link to="/" className={styles.navLink}>返回首页</Link>
        </div>
      </div>
    </div>
  )
}

export default Settings
