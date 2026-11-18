import { useRoute } from './router';
import { Home } from './screens/Home';
import { Camera } from './screens/Camera';
import { Result } from './screens/Result';
import { Players } from './screens/Players';
import { Stats } from './screens/Stats';
import { SessionSetup, SessionView } from './screens/Session';
import { Settings } from './screens/Settings';
import { Help } from './screens/Help';

export function App() {
  const [page, arg] = useRoute();
  switch (page) {
    case 'camera': return <Camera />;
    case 'result': return <Result />;
    case 'players': return <Players />;
    case 'stats': return <Stats />;
    case 'session': return arg === 'new' ? <SessionSetup /> : <SessionView id={arg} />;
    case 'settings': return <Settings />;
    case 'help': return <Help />;
    default: return <Home />;
  }
}
