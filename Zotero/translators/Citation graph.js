{
	"translatorID": "19afa3fd-1c7f-4eb8-a37e-8d07768493e8",
	"label": "Citation graph",
	"description": "exports a citation graph in graphml format. Use gephi or yEd to clean up and visualize",
	"creator": "Emiliano heyns",
	"target": "dot",
	"minVersion": "4.0.27",
	"maxVersion": "",
	"translatorType": 2,
	"browserSupport": "gcsv",
	"inRepository": false,
	"displayOptions": {
		"Title": false,
		"Authors": false,
		"Year": false
	},
	"priority": 100,
	"configOptions": {
		"hash": "66b478bf62a57ffff2e963e336977ced696b3ef252fe10a93257e30594e23f7e"
	},
	"lastUpdated": "2022-05-05"
}

ZOTERO_CONFIG = {"GUID":"zotero@chnm.gmu.edu","ID":"zotero","CLIENT_NAME":"Zotero","DOMAIN_NAME":"zotero.org","PRODUCER":"Digital Scholar","PRODUCER_URL":"https://digitalscholar.org","REPOSITORY_URL":"https://repo.zotero.org/repo/","BASE_URI":"http://zotero.org/","WWW_BASE_URL":"https://www.zotero.org/","PROXY_AUTH_URL":"https://zoteroproxycheck.s3.amazonaws.com/test","API_URL":"https://api.zotero.org/","STREAMING_URL":"wss://stream.zotero.org/","SERVICES_URL":"https://services.zotero.org/","API_VERSION":3,"CONNECTOR_MIN_VERSION":"5.0.39","PREF_BRANCH":"extensions.zotero.","BOOKMARKLET_ORIGIN":"https://www.zotero.org","BOOKMARKLET_URL":"https://www.zotero.org/bookmarklet/","START_URL":"https://www.zotero.org/start","QUICK_START_URL":"https://www.zotero.org/support/quick_start_guide","PDF_TOOLS_URL":"https://www.zotero.org/download/xpdf/","SUPPORT_URL":"https://www.zotero.org/support/","SYNC_INFO_URL":"https://www.zotero.org/support/sync","TROUBLESHOOTING_URL":"https://www.zotero.org/support/getting_help","FEEDBACK_URL":"https://forums.zotero.org/","CONNECTORS_URL":"https://www.zotero.org/download/connectors","CHANGELOG_URL":"https://www.zotero.org/support/changelog","CREDITS_URL":"https://www.zotero.org/support/credits_and_acknowledgments","LICENSING_URL":"https://www.zotero.org/support/licensing","GET_INVOLVED_URL":"https://www.zotero.org/getinvolved","DICTIONARIES_URL":"https://download.zotero.org/dictionaries/"}
if (typeof ZOTERO_TRANSLATOR_INFO === 'undefined') var ZOTERO_TRANSLATOR_INFO = {"translatorID":"19afa3fd-1c7f-4eb8-a37e-8d07768493e8","label":"Citation graph","description":"exports a citation graph in graphml format. Use gephi or yEd to clean up and visualize","creator":"Emiliano heyns","target":"dot","minVersion":"4.0.27","maxVersion":"","translatorType":2,"browserSupport":"gcsv","inRepository":false,"displayOptions":{"Title":false,"Authors":false,"Year":false},"priority":100};
var Citationgraph__Translator__doExport = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined")
      return require.apply(this, arguments);
    throw new Error('Dynamic require of "' + x + '" is not supported');
  });
  var __esm = (fn, res) => function __init() {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  };
  var __commonJS = (cb, mod) => function __require2() {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  };
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // node_modules/process/browser.js
  var require_browser = __commonJS({
    "node_modules/process/browser.js"(exports, module) {
      init_globals();
      var process2 = module.exports = {};
      var cachedSetTimeout;
      var cachedClearTimeout;
      function defaultSetTimout() {
        throw new Error("setTimeout has not been defined");
      }
      function defaultClearTimeout() {
        throw new Error("clearTimeout has not been defined");
      }
      (function() {
        try {
          if (typeof setTimeout === "function") {
            cachedSetTimeout = setTimeout;
          } else {
            cachedSetTimeout = defaultSetTimout;
          }
        } catch (e) {
          cachedSetTimeout = defaultSetTimout;
        }
        try {
          if (typeof clearTimeout === "function") {
            cachedClearTimeout = clearTimeout;
          } else {
            cachedClearTimeout = defaultClearTimeout;
          }
        } catch (e) {
          cachedClearTimeout = defaultClearTimeout;
        }
      })();
      function runTimeout(fun) {
        if (cachedSetTimeout === setTimeout) {
          return setTimeout(fun, 0);
        }
        if ((cachedSetTimeout === defaultSetTimout || !cachedSetTimeout) && setTimeout) {
          cachedSetTimeout = setTimeout;
          return setTimeout(fun, 0);
        }
        try {
          return cachedSetTimeout(fun, 0);
        } catch (e) {
          try {
            return cachedSetTimeout.call(null, fun, 0);
          } catch (e2) {
            return cachedSetTimeout.call(this, fun, 0);
          }
        }
      }
      function runClearTimeout(marker) {
        if (cachedClearTimeout === clearTimeout) {
          return clearTimeout(marker);
        }
        if ((cachedClearTimeout === defaultClearTimeout || !cachedClearTimeout) && clearTimeout) {
          cachedClearTimeout = clearTimeout;
          return clearTimeout(marker);
        }
        try {
          return cachedClearTimeout(marker);
        } catch (e) {
          try {
            return cachedClearTimeout.call(null, marker);
          } catch (e2) {
            return cachedClearTimeout.call(this, marker);
          }
        }
      }
      var queue = [];
      var draining = false;
      var currentQueue;
      var queueIndex = -1;
      function cleanUpNextTick() {
        if (!draining || !currentQueue) {
          return;
        }
        draining = false;
        if (currentQueue.length) {
          queue = currentQueue.concat(queue);
        } else {
          queueIndex = -1;
        }
        if (queue.length) {
          drainQueue();
        }
      }
      function drainQueue() {
        if (draining) {
          return;
        }
        var timeout = runTimeout(cleanUpNextTick);
        draining = true;
        var len = queue.length;
        while (len) {
          currentQueue = queue;
          queue = [];
          while (++queueIndex < len) {
            if (currentQueue) {
              currentQueue[queueIndex].run();
            }
          }
          queueIndex = -1;
          len = queue.length;
        }
        currentQueue = null;
        draining = false;
        runClearTimeout(timeout);
      }
      process2.nextTick = function(fun) {
        var args = new Array(arguments.length - 1);
        if (arguments.length > 1) {
          for (var i = 1; i < arguments.length; i++) {
            args[i - 1] = arguments[i];
          }
        }
        queue.push(new Item(fun, args));
        if (queue.length === 1 && !draining) {
          runTimeout(drainQueue);
        }
      };
      function Item(fun, array) {
        this.fun = fun;
        this.array = array;
      }
      Item.prototype.run = function() {
        this.fun.apply(null, this.array);
      };
      process2.title = "browser";
      process2.browser = true;
      process2.env = {};
      process2.argv = [];
      process2.version = "";
      process2.versions = {};
      function noop() {
      }
      process2.on = noop;
      process2.addListener = noop;
      process2.once = noop;
      process2.off = noop;
      process2.removeListener = noop;
      process2.removeAllListeners = noop;
      process2.emit = noop;
      process2.prependListener = noop;
      process2.prependOnceListener = noop;
      process2.listeners = function(name) {
        return [];
      };
      process2.binding = function(name) {
        throw new Error("process.binding is not supported");
      };
      process2.cwd = function() {
        return "/";
      };
      process2.chdir = function(dir) {
        throw new Error("process.chdir is not supported");
      };
      process2.umask = function() {
        return 0;
      };
    }
  });

  // setup/loaders/globals.js
  var process, global;
  var init_globals = __esm({
    "setup/loaders/globals.js"() {
      process = require_browser();
      global = Function("return this")();
    }
  });

  // translators/Citation graph.ts
  var Citation_graph_exports = {};
  __export(Citation_graph_exports, {
    Translator: () => Translator,
    doExport: () => doExport
  });
  init_globals();

  // translators/lib/translator.ts
  init_globals();

  // gen/preferences/meta.ts
  init_globals();
  var names = [
    "ascii",
    "asciiBibLaTeX",
    "asciiBibTeX",
    "autoAbbrev",
    "autoAbbrevStyle",
    "autoExport",
    "autoExportDelay",
    "autoExportIdleWait",
    "autoExportPathReplaceDiacritics",
    "autoExportPathReplaceDirSep",
    "autoExportPathReplaceSpace",
    "automaticTags",
    "autoPinDelay",
    "auxImport",
    "baseAttachmentPath",
    "biblatexExtendedDateFormat",
    "biblatexExtendedNameFormat",
    "biblatexExtractEprint",
    "bibtexParticleNoOp",
    "bibtexURL",
    "cache",
    "cacheFlushInterval",
    "charmap",
    "citeCommand",
    "citekeyFold",
    "citekeyFormat",
    "citekeyFormatBackup",
    "citekeySearch",
    "csquotes",
    "DOIandURL",
    "exportBibTeXStrings",
    "exportBraceProtection",
    "exportTitleCase",
    "extraMergeCitekeys",
    "extraMergeCSL",
    "extraMergeTeX",
    "git",
    "import",
    "importBibTeXStrings",
    "importCaseProtection",
    "importCitationKey",
    "importExtra",
    "importJabRefAbbreviations",
    "importJabRefStrings",
    "importSentenceCase",
    "importUnknownTexCommand",
    "itemObserverDelay",
    "jabrefFormat",
    "jieba",
    "keyConflictPolicy",
    "keyScope",
    "kuroshiro",
    "mapMath",
    "mapText",
    "mapUnicode",
    "parseParticles",
    "patchDates",
    "platform",
    "postscript",
    "postscriptOverride",
    "preferencesOverride",
    "qualityReport",
    "quickCopyEta",
    "quickCopyMode",
    "quickCopyOrgMode",
    "quickCopyPandocBrackets",
    "quickCopySelectLink",
    "rawImports",
    "rawLaTag",
    "relativeFilePaths",
    "retainCache",
    "scrubDatabase",
    "separatorList",
    "separatorNames",
    "skipFields",
    "skipWords",
    "startupProgress",
    "strings",
    "stringsOverride",
    "testing",
    "verbatimFields",
    "warnBulkModify",
    "warnTitleCased",
    "workers"
  ];
  var affects = {
    ascii: ["Better BibLaTeX", "Better BibTeX"],
    asciiBibLaTeX: ["Better BibLaTeX"],
    asciiBibTeX: ["Better BibTeX"],
    autoAbbrev: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    autoAbbrevStyle: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    automaticTags: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    baseAttachmentPath: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    biblatexExtendedDateFormat: ["Better BibLaTeX"],
    biblatexExtendedNameFormat: ["Better BibLaTeX"],
    biblatexExtractEprint: ["Better BibLaTeX", "Better BibTeX"],
    bibtexParticleNoOp: ["Better BibTeX"],
    bibtexURL: ["Better BibTeX"],
    cache: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    charmap: ["Better BibLaTeX", "Better BibTeX"],
    csquotes: ["Better BibLaTeX", "Better BibTeX"],
    DOIandURL: ["Better BibLaTeX", "Better BibTeX"],
    exportBibTeXStrings: ["Better BibLaTeX", "Better BibTeX"],
    exportBraceProtection: ["Better BibLaTeX", "Better BibTeX"],
    exportTitleCase: ["Better BibLaTeX", "Better BibTeX"],
    jabrefFormat: ["Better BibLaTeX", "Better BibTeX"],
    mapMath: ["Better BibLaTeX", "Better BibTeX"],
    mapText: ["Better BibLaTeX", "Better BibTeX"],
    mapUnicode: ["Better BibLaTeX", "Better BibTeX"],
    parseParticles: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    postscript: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    qualityReport: ["Better BibLaTeX", "Better BibTeX"],
    rawLaTag: ["Better BibLaTeX", "Better BibTeX"],
    relativeFilePaths: ["Better BibLaTeX", "Better BibTeX"],
    separatorList: ["Better BibLaTeX", "Better BibTeX"],
    separatorNames: ["Better BibLaTeX", "Better BibTeX"],
    skipFields: ["Better BibLaTeX", "Better BibTeX", "Better CSL JSON", "Better CSL YAML"],
    skipWords: ["Better BibLaTeX", "Better BibTeX"],
    strings: ["Better BibLaTeX", "Better BibTeX"]
  };
  var defaults = {
    ascii: "",
    asciiBibLaTeX: false,
    asciiBibTeX: true,
    autoAbbrev: false,
    autoAbbrevStyle: "",
    autoExport: "immediate",
    autoExportDelay: 5,
    autoExportIdleWait: 10,
    autoExportPathReplaceDiacritics: false,
    autoExportPathReplaceDirSep: "-",
    autoExportPathReplaceSpace: " ",
    automaticTags: true,
    autoPinDelay: 0,
    auxImport: false,
    baseAttachmentPath: "",
    biblatexExtendedDateFormat: true,
    biblatexExtendedNameFormat: false,
    biblatexExtractEprint: true,
    bibtexParticleNoOp: false,
    bibtexURL: "off",
    cache: true,
    cacheFlushInterval: 5,
    charmap: "",
    citeCommand: "cite",
    citekeyFold: true,
    citekeyFormat: "auth.lower + shorttitle(3,3) + year",
    citekeyFormatBackup: "",
    citekeySearch: true,
    csquotes: "",
    DOIandURL: "both",
    exportBibTeXStrings: "off",
    exportBraceProtection: true,
    exportTitleCase: true,
    extraMergeCitekeys: false,
    extraMergeCSL: false,
    extraMergeTeX: false,
    git: "config",
    import: true,
    importBibTeXStrings: true,
    importCaseProtection: "as-needed",
    importCitationKey: true,
    importExtra: true,
    importJabRefAbbreviations: true,
    importJabRefStrings: true,
    importSentenceCase: "on+guess",
    importUnknownTexCommand: "ignore",
    itemObserverDelay: 5,
    jabrefFormat: 0,
    jieba: false,
    keyConflictPolicy: "keep",
    keyScope: "library",
    kuroshiro: false,
    mapMath: "",
    mapText: "",
    mapUnicode: "conservative",
    parseParticles: true,
    patchDates: "",
    platform: "",
    postscript: "",
    postscriptOverride: "",
    preferencesOverride: "",
    qualityReport: false,
    quickCopyEta: "",
    quickCopyMode: "latex",
    quickCopyOrgMode: "zotero",
    quickCopyPandocBrackets: false,
    quickCopySelectLink: "zotero",
    rawImports: false,
    rawLaTag: "#LaTeX",
    relativeFilePaths: false,
    retainCache: false,
    scrubDatabase: false,
    separatorList: "and",
    separatorNames: "and",
    skipFields: "",
    skipWords: "a,ab,aboard,about,above,across,after,against,al,along,amid,among,an,and,anti,around,as,at,before,behind,below,beneath,beside,besides,between,beyond,but,by,d,da,das,de,del,dell,dello,dei,degli,della,dell,delle,dem,den,der,des,despite,die,do,down,du,during,ein,eine,einem,einen,einer,eines,el,en,et,except,for,from,gli,i,il,in,inside,into,is,l,la,las,le,les,like,lo,los,near,nor,of,off,on,onto,or,over,past,per,plus,round,save,since,so,some,sur,than,the,through,to,toward,towards,un,una,unas,under,underneath,une,unlike,uno,unos,until,up,upon,versus,via,von,while,with,within,without,yet,zu,zum",
    startupProgress: "popup",
    strings: "",
    stringsOverride: "",
    testing: false,
    verbatimFields: "url,doi,file,ids,eprint,verba,verbb,verbc,groups",
    warnBulkModify: 10,
    warnTitleCased: false,
    workers: 1
  };
  var schema = {
    autoExport: {
      preferences: ["asciiBibLaTeX", "asciiBibTeX", "biblatexExtendedNameFormat", "bibtexParticleNoOp", "bibtexURL", "DOIandURL"],
      displayOptions: ["exportNotes", "useJournalAbbreviation"]
    },
    translator: {
      "Better BibTeX": {
        autoexport: true,
        cached: true,
        preferences: ["asciiBibTeX", "bibtexParticleNoOp", "bibtexURL", "DOIandURL"],
        displayOptions: ["exportNotes", "useJournalAbbreviation"],
        types: {
          asciiBibTeX: { type: "boolean" },
          bibtexParticleNoOp: { type: "boolean" },
          bibtexURL: { enum: ["off", "note", "note-url-ish", "url", "url-ish"] },
          DOIandURL: { enum: ["both", "doi", "url"] },
          exportNotes: { type: "boolean" },
          useJournalAbbreviation: { type: "boolean" }
        }
      },
      "Better CSL JSON": {
        autoexport: true,
        cached: true,
        preferences: [],
        displayOptions: [],
        types: {}
      },
      "BetterBibTeX JSON": {
        autoexport: true,
        cached: false,
        preferences: [],
        displayOptions: ["exportNotes"],
        types: {
          exportNotes: { type: "boolean" }
        }
      },
      "Better BibLaTeX": {
        autoexport: true,
        cached: true,
        preferences: ["asciiBibLaTeX", "biblatexExtendedNameFormat", "DOIandURL"],
        displayOptions: ["exportNotes", "useJournalAbbreviation"],
        types: {
          asciiBibLaTeX: { type: "boolean" },
          biblatexExtendedNameFormat: { type: "boolean" },
          DOIandURL: { enum: ["both", "doi", "url"] },
          exportNotes: { type: "boolean" },
          useJournalAbbreviation: { type: "boolean" }
        }
      },
      "Better CSL YAML": {
        autoexport: true,
        cached: true,
        preferences: [],
        displayOptions: [],
        types: {}
      }
    }
  };

  // content/client.ts
  init_globals();
  if (typeof Components !== "undefined")
    Components.utils.import("resource://zotero/config.js");
  var client = typeof Zotero !== "undefined" ? ZOTERO_CONFIG.GUID.replace(/@.*/, "").replace("-", "") : "zotero";

  // content/environment.ts
  init_globals();
  var environment = {
    node: typeof process === "object" && typeof __require === "function" && typeof importScripts !== "function",
    worker: typeof importScripts === "function",
    zotero: typeof Components !== "undefined",
    name: ""
  };
  environment.name = Object.entries(environment).map(([name, on]) => on ? name : "").filter((name) => name).join("/");

  // content/ping.ts
  init_globals();
  var Pinger = class {
    constructor({ start = 0, total, step = 5, name = "", callback }) {
      this.incr = 100 / total;
      this.name = name;
      this.pct = start * this.incr;
      this.step = step;
      this.callback = callback;
      this.next = Math.floor(this.pct / step) * step;
      if (this.name)
        Zotero.debug(`ping: ${name} start ${JSON.stringify({ ...this, start, total })}`);
      this.emit();
    }
    update() {
      this.pct += this.incr;
      if (this.name)
        Zotero.debug(`ping: ${this.name} update to ${this.pct}`);
      if (Math.round(this.pct) >= this.next)
        this.emit();
    }
    emit() {
      if (this.callback) {
        if (this.name)
          Zotero.debug(`ping: ${this.name} emit ${Math.min(this.next, 100)}`);
        this.callback(Math.min(this.next, 100));
        if (this.next > 100)
          this.callback = null;
        this.next += this.step;
      }
    }
    done() {
      if (this.name)
        Zotero.debug(`ping: ${this.name} done`);
      if (this.callback && this.pct < this.next)
        this.callback(Math.min(this.next, 100));
    }
  };

  // translators/lib/translator.ts
  var cacheDisabler = new class {
    get(target, property) {
      if (property === "collections") {
        target.$cacheable = false;
      }
      return target[property];
    }
  }();
  var Items = class {
    constructor(cacheable) {
      this.list = [];
      this.map = {};
      let item;
      while (item = Zotero.nextItem()) {
        item.$cacheable = cacheable;
        item.journalAbbreviation = item.journalAbbreviation || item.autoJournalAbbreviation;
        this.list.push(this.map[item.itemID] = this.map[item.itemKey] = new Proxy(item, cacheDisabler));
      }
      this.list.sort((a, b) => {
        const ka = [a.citationKey || a.itemType, a.dateModified || a.dateAdded, a.itemID].join("	");
        const kb = [b.citationKey || b.itemType, b.dateModified || b.dateAdded, b.itemID].join("	");
        return ka.localeCompare(kb, void 0, { sensitivity: "base" });
      });
      this.ping = new Pinger({
        total: this.list.length,
        callback: (pct) => environment.worker ? Zotero.BetterBibTeX.setProgress(pct) : null
      });
    }
    *items() {
      for (const item of this.list) {
        yield this.current = item;
        this.ping.update();
      }
      this.ping.done();
    }
    *regularitems() {
      for (const item of this.list) {
        switch (item.itemType) {
          case "annotation":
          case "note":
          case "attachment":
            break;
          default:
            yield this.current = item;
        }
        this.ping.update();
      }
      this.ping.done();
    }
  };
  function escapeRegExp(text) {
    return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&");
  }
  var ITranslator = class {
    constructor() {
      this.export = {
        dir: void 0,
        path: void 0
      };
      this.cacheable = true;
      this.initialized = false;
      this.header = ZOTERO_TRANSLATOR_INFO;
      this[this.header.label.replace(/[^a-z]/ig, "")] = true;
      this.BetterTeX = this.BetterBibTeX || this.BetterBibLaTeX;
      this.BetterCSL = this.BetterCSLJSON || this.BetterCSLYAML;
      this.options = this.header.displayOptions || {};
      const collator = new Intl.Collator("en");
      this.stringCompare = collator.compare.bind(collator);
    }
    get exportDir() {
      this._items.current.$cacheable = false;
      return this.export.dir;
    }
    get exportPath() {
      this._items.current.$cacheable = false;
      return this.export.path;
    }
    typefield(field) {
      field = field.trim();
      if (field.startsWith("bibtex."))
        return this.BetterBibTeX ? field.replace(/^bibtex\./, "") : "";
      if (field.startsWith("biblatex."))
        return this.BetterBibLaTeX ? field.replace(/^biblatex\./, "") : "";
      return field;
    }
    init(mode) {
      var _a, _b, _c, _d, _e;
      this.platform = Zotero.getHiddenPref("better-bibtex.platform");
      this.isJurisM = client === "jurism";
      this.isZotero = !this.isJurisM;
      this.paths = {
        caseSensitive: this.platform !== "mac" && this.platform !== "win",
        sep: this.platform === "win" ? "\\" : "/"
      };
      try {
        if (Zotero.getOption("cache") === false)
          this.cacheable = false;
      } catch (err) {
      }
      for (const key in this.options) {
        if (typeof this.options[key] === "boolean") {
          this.options[key] = Zotero.getOption(key);
        } else {
          this.options[key] = !!Zotero.getOption(key);
        }
      }
      if (mode === "export") {
        this.cache = {
          hits: 0,
          requests: 0
        };
        this.export = {
          dir: Zotero.getOption("exportDir"),
          path: Zotero.getOption("exportPath")
        };
        if ((_a = this.export.dir) == null ? void 0 : _a.endsWith(this.paths.sep))
          this.export.dir = this.export.dir.slice(0, -1);
        this.options.cacheUse = Zotero.getOption("cacheUse");
      }
      this.preferences = Object.entries(defaults).reduce((acc, [pref, dflt]) => {
        var _a2, _b2;
        acc[pref] = (_b2 = (_a2 = this.getPreferenceOverride(pref)) != null ? _a2 : Zotero.getHiddenPref(`better-bibtex.${pref}`)) != null ? _b2 : dflt;
        return acc;
      }, {});
      this.skipFields = this.preferences.skipFields.toLowerCase().split(",").map((field) => this.typefield(field)).filter((s) => s);
      this.skipField = this.skipFields.reduce((acc, field) => {
        acc[field] = true;
        return acc;
      }, {});
      this.verbatimFields = this.preferences.verbatimFields.toLowerCase().split(",").map((field) => this.typefield(field)).filter((s) => s);
      if (!this.verbatimFields.length)
        this.verbatimFields = null;
      this.csquotes = this.preferences.csquotes ? { open: this.preferences.csquotes[0], close: this.preferences.csquotes[1] } : null;
      this.preferences.testing = Zotero.getHiddenPref("better-bibtex.testing");
      if (mode === "export") {
        this.unicode = !this.preferences[`ascii${this.header.label.replace(/Better /, "")}`];
        if (this.preferences.baseAttachmentPath && (this.export.dir === this.preferences.baseAttachmentPath || ((_b = this.export.dir) == null ? void 0 : _b.startsWith(this.preferences.baseAttachmentPath + this.paths.sep)))) {
          this.preferences.relativeFilePaths = true;
        }
        this.cacheable = this.cacheable && this.preferences.cache && !(this.options.exportFileData || this.preferences.relativeFilePaths || this.preferences.baseAttachmentPath && ((_c = this.export.dir) == null ? void 0 : _c.startsWith(this.preferences.baseAttachmentPath)));
        if (this.BetterTeX) {
          this.preferences.separatorList = this.preferences.separatorList.trim();
          this.preferences.separatorNames = this.preferences.separatorNames.trim();
          this.and = {
            list: {
              re: new RegExp(escapeRegExp(this.preferences.separatorList), "g"),
              repl: ` {${this.preferences.separatorList}} `
            },
            names: {
              re: new RegExp(` ${escapeRegExp(this.preferences.separatorNames)} `, "g"),
              repl: ` {${this.preferences.separatorNames}} `
            }
          };
          this.preferences.separatorList = ` ${this.preferences.separatorList} `;
          this.preferences.separatorNames = ` ${this.preferences.separatorNames} `;
        }
      }
      this.collections = {};
      if (mode === "export" && ((_d = this.header.configOptions) == null ? void 0 : _d.getCollections) && Zotero.nextCollection) {
        let collection;
        while (collection = Zotero.nextCollection()) {
          this.registerCollection(collection, "");
        }
      }
      if (!this.initialized && mode === "export" && this.preferences.testing && typeof __estrace === "undefined" && ((_e = schema.translator[this.header.label]) == null ? void 0 : _e.cached)) {
        const ignored = ["testing"];
        this.preferences = new Proxy(this.preferences, {
          set: (object, property, _value) => {
            throw new TypeError(`Unexpected set of preference ${String(property)}`);
          },
          get: (object, property) => {
            var _a2;
            if (property === "toJSON")
              return object[property];
            if (!names.includes(property))
              throw new TypeError(`Unsupported preference ${property}`);
            if (!ignored.includes(property) && !((_a2 = affects[property]) == null ? void 0 : _a2.includes(this.header.label)))
              throw new TypeError(`Preference ${property} claims not to affect ${this.header.label}`);
            return object[property];
          }
        });
      }
      this.initialized = true;
    }
    getPreferenceOverride(pref) {
      try {
        const override = Zotero.getOption(`preference_${pref}`);
        if (typeof override !== "undefined")
          this.cacheable = false;
        return override;
      } catch (err) {
        return void 0;
      }
    }
    registerCollection(collection, parent) {
      const key = (collection.primary ? collection.primary : collection).key;
      const children = collection.children || collection.descendents || [];
      const collections = children.filter((coll) => coll.type === "collection");
      this.collections[key] = {
        key,
        parent,
        name: collection.name,
        collections: collections.map((coll) => coll.key),
        items: children.filter((coll) => coll.type === "item").map((item) => item.id)
      };
      for (collection of collections) {
        this.registerCollection(collection, key);
      }
    }
    get collectionTree() {
      return Object.values(this.collections).filter((coll) => !coll.parent).map((coll) => this.nestedCollection(coll));
    }
    nestedCollection(collection) {
      this._items = this._items || new Items(this.cacheable);
      const nested = {
        key: collection.key,
        name: collection.name,
        items: collection.items.map((itemID) => this._items.map[itemID]).filter((item) => item),
        collections: collection.collections.map((key) => this.nestedCollection(this.collections[key])).filter((coll) => coll)
      };
      for (const coll of nested.collections) {
        coll.parent = nested;
      }
      return nested;
    }
    get items() {
      this._items = this._items || new Items(this.cacheable);
      return this._items.items();
    }
    get regularitems() {
      this._items = this._items || new Items(this.cacheable);
      return this._items.regularitems();
    }
  };
  var Translator = new ITranslator();

  // translators/Citation graph.ts
  function node(id, attributes = {}) {
    let n = JSON.stringify(id);
    const attrs = Object.entries(attributes).map(([key, value]) => `${key}=${JSON.stringify(value)}`).join(", ");
    if (attrs)
      n += ` [${attrs}]`;
    Zotero.write(`  ${n};
`);
  }
  function edge(source, target, attributes = {}) {
    let e = `${JSON.stringify(source)} -> ${JSON.stringify(target)}`;
    const attrs = Object.entries(attributes).map(([key, value]) => `${key}=${JSON.stringify(value)}`).join(", ");
    if (attrs)
      e += ` [${attrs}]`;
    Zotero.write(`  ${e};
`);
  }
  function doExport() {
    var _a, _b;
    Translator.init("export");
    Zotero.write("digraph CitationGraph {\n");
    Zotero.write("  concentrate=true;\n");
    const add = {
      title: Zotero.getOption("Title"),
      authors: Zotero.getOption("Authors"),
      year: Zotero.getOption("Year")
    };
    const items = [];
    for (const ref of Translator.regularitems) {
      const label = [ref.citationKey];
      if (add.title && ref.title) {
        label.push(`\u201C${ref.title.replace(/"/g, "'")}\u201D`);
      }
      const author = [];
      if (add.authors && ref.creators && ref.creators.length) {
        const name = (_a = ref.creators) == null ? void 0 : _a.map((creator) => (creator.name || creator.lastName || "").replace(/"/g, "'")).filter((creator) => creator).join(", ");
        if (name)
          author.push(name);
      }
      if (add.year && ref.date) {
        let date = Zotero.BetterBibTeX.parseDate(ref.date);
        if (date.from)
          date = date.from;
        if (date.year)
          author.push(`(${date.year})`);
      }
      if (author.length)
        label.push(author.join(" "));
      items.push({
        id: `node-${ref.uri.replace(/.*\//, "")}`,
        label: label.join("\n"),
        relations: ((_b = ref.relations) == null ? void 0 : _b["dc:relation"]) || [],
        cites: [].concat.apply([], (ref.extra || "").split("\n").filter((line) => line.startsWith("cites:")).map((line) => line.replace(/^cites:/, "").trim()).filter((keys) => keys).map((keys) => keys.split(/\s*,\s*/))),
        citationKey: ref.citationKey,
        uri: ref.uri
      });
    }
    for (const item of items) {
      node(item.id, { label: item.label });
      for (const uri of item.relations) {
        const other = items.find((o) => o.uri === uri);
        if (other) {
          edge(item.id, other.id);
        } else {
          edge(item.id, uri.replace(/.*\//, ""), { style: "dashed", dir: "both" });
        }
      }
      for (const citationKey of item.cites) {
        const other = items.find((o) => o.citationKey === citationKey);
        if (other) {
          edge(item.id, other.id);
        } else {
          edge(item.id, citationKey, { style: "dashed" });
        }
      }
    }
    Zotero.write("}");
  }
  return __toCommonJS(Citation_graph_exports);
})();
var { Translator, doExport } = Citationgraph__Translator__doExport;
