import vm from "node:vm";

export function hookHarness(
  compiled,
  componentName,
  imports = {},
  globals = {},
) {
  const slots = [];
  const timers = new Map();
  let cursor = 0;
  let dirty = false;
  let mounted = true;
  let writesAfterUnmount = 0;
  let pendingEffects = [];
  let props;
  let tree;
  const changed = (before, after) =>
    !before ||
    !after ||
    before.length !== after.length ||
    after.some((value, i) => !Object.is(value, before[i]));
  const hooks = {
    useState(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = initial;
      return [
        slots[index],
        (value) => {
          if (!mounted) writesAfterUnmount++;
          slots[index] =
            typeof value === "function" ? value(slots[index]) : value;
          dirty = true;
        },
      ];
    },
    useRef(initial) {
      const index = cursor++;
      if (!(index in slots)) slots[index] = { current: initial };
      return slots[index];
    },
    useCallback(callback, dependencies) {
      const index = cursor++;
      if (changed(slots[index]?.dependencies, dependencies)) {
        slots[index] = { callback, dependencies };
      }
      return slots[index].callback;
    },
    useEffect(effect, dependencies) {
      const index = cursor++;
      if (changed(slots[index]?.dependencies, dependencies)) {
        pendingEffects.push(() => {
          slots[index]?.cleanup?.();
          slots[index] = { dependencies, cleanup: effect() };
        });
      }
    },
  };
  const exports = {};
  vm.runInNewContext(compiled, {
    Error,
    AbortController,
    Blob,
    ...globals,
    exports,
    require(name) {
      if (name === "react") return hooks;
      if (name === "react/jsx-runtime")
        return {
          jsx: (type, props) => ({ type, props }),
          jsxs: (type, props) => ({ type, props }),
        };
      return imports[name] || {};
    },
    setTimeout(callback, delay) {
      const id = Symbol();
      timers.set(id, { callback, delay });
      return id;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
  });
  function render(nextProps = props) {
    props = nextProps;
    do {
      dirty = false;
      cursor = 0;
      tree = exports[componentName](props);
      const effects = pendingEffects;
      pendingEffects = [];
      effects.forEach((effect) => effect());
    } while (dirty);
    return tree;
  }
  return {
    render,
    async settle() {
      for (let i = 0; i < 12; i++) await Promise.resolve();
      return render();
    },
    nodes(type) {
      const matches = [];
      function visit(node) {
        if (!node || typeof node !== "object") return;
        if (node.type === type || node.props?.role === type) matches.push(node);
        const children = node.props?.children;
        (Array.isArray(children) ? children : [children]).forEach(visit);
      }
      visit(tree);
      return matches;
    },
    get timers() {
      return timers;
    },
    get writesAfterUnmount() {
      return writesAfterUnmount;
    },
    unmount() {
      mounted = false;
      slots.forEach((slot) => slot?.cleanup?.());
    },
  };
}
