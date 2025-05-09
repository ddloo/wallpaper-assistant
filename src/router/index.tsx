import { createBrowserRouter, RouteObject } from 'react-router-dom'
import MainLayout from '../layouts/MainLayout'
import Home from '../pages/home'
import Settings from '../pages/settings'

const routes: RouteObject[] = [
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: <Home />
      },
      {
        path: 'settings',
        element: <Settings />
      }
    ]
  }
]

export const router = createBrowserRouter(routes)
