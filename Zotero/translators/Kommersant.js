{
	"translatorID": "43bc17ed-e994-4fdb-ac28-594c839658ca",
	"label": "Kommersant",
	"creator": "Avram Lyon",
	"target": "^https?://(www\\.)?kommersant\\.ru/",
	"minVersion": "2.1",
	"maxVersion": "",
	"priority": 100,
	"inRepository": true,
	"translatorType": 4,
	"browserSupport": "gcsibv",
	"lastUpdated": "2017-01-01 16:02:40"
}

/*********************** BEGIN FRAMEWORK ***********************/
/**
    Copyright (c) 2010-2013, Erik Hetzner

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU Affero General Public License
    as published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public
    License along with this program.  If not, see
    <http://www.gnu.org/licenses/>.
*/

/**
 * Flatten a nested array; e.g., [[1], [2,3]] -> [1,2,3]
 */
function flatten(a) {
    var retval = new Array();
    for (var i in a) {
        var entry = a[i];
        if (entry instanceof Array) {
            retval = retval.concat(flatten(entry));
        } else {
            retval.push(entry);
        }
    }
    return retval;
}

var FW = {
    _scrapers : new Array()
};

FW._Base = function () {
    this.callHook = function (hookName, item, doc, url) {
        if (typeof this['hooks'] === 'object') {
            var hook = this['hooks'][hookName];
            if (typeof hook === 'function') {
                hook(item, doc, url);
            }
        }
    };

    this.evaluateThing = function(val, doc, url) {
        var valtype = typeof val;
        if (valtype === 'object') {
            if (val instanceof Array) {
                /* map over each array val */
                /* this.evaluate gets out of scope */
                var parentEval = this.evaluateThing;
                var retval = val.map ( function(i) { return parentEval (i, doc, url); } );
                return flatten(retval);
            } else {
                return val.evaluate(doc, url);
            }
        } else if (valtype === 'function') {
            return val(doc, url);
        } else {
            return val;
        }
    };

    /*
     * makeItems is the function that does the work of making an item.
     * doc: the doc tree for the item
     * url: the url for the item
     * attachments ...
     * eachItem: a function to be called for each item made, with the arguments (doc, url, ...)
     * ret: the function to call when you are done, with no args
     */
    this.makeItems = function (doc, url, attachments, eachItem, ret) {
        ret();
    }

};

FW.Scraper = function (init) { 
    FW._scrapers.push(new FW._Scraper(init));
};

FW._Scraper = function (init) {
    for (x in init) {
        this[x] = init[x];
    }

    this._singleFieldNames = [
        "abstractNote",
        "applicationNumber",
        "archive",
        "archiveLocation",
        "artworkMedium",
        "artworkSize",
        "assignee",
        "audioFileType",
        "audioRecordingType",
        "billNumber",
        "blogTitle",
        "bookTitle",
        "callNumber",
        "caseName",
        "code",
        "codeNumber",
        "codePages",
        "codeVolume",
        "committee",
        "company",
        "conferenceName",
        "country",
        "court",
        "date",
        "dateDecided",
        "dateEnacted",
        "dictionaryTitle",
        "distributor",
        "docketNumber",
        "documentNumber",
        "DOI",
        "edition",
        "encyclopediaTitle",
        "episodeNumber",
        "extra",
        "filingDate",
        "firstPage",
        "forumTitle",
        "genre",
        "history",
        "institution",
        "interviewMedium",
        "ISBN",
        "ISSN",
        "issue",
        "issueDate",
        "issuingAuthority",
        "journalAbbreviation",
        "label",
        "language",
        "legalStatus",
        "legislativeBody",
        "letterType",
        "libraryCatalog",
        "manuscriptType",
        "mapType",
        "medium",
        "meetingName",
        "nameOfAct",
        "network",
        "number",
        "numberOfVolumes",
        "numPages",
        "pages",
        "patentNumber",
        "place",
        "postType",
        "presentationType",
        "priorityNumbers",
        "proceedingsTitle",
        "programTitle",
        "programmingLanguage",
        "publicLawNumber",
        "publicationTitle",
        "publisher",
        "references",
        "reportNumber",
        "reportType",
        "reporter",
        "reporterVolume",
        "rights",
        "runningTime",
        "scale",
        "section",
        "series",
        "seriesNumber",
        "seriesText",
        "seriesTitle",
        "session",
        "shortTitle",
        "studio",
        "subject",
        "system",
        "thesisType",
        "title",
        "type",
        "university",
        "url",
        "version",
        "videoRecordingType",
        "volume",
        "websiteTitle",
        "websiteType" ];
    
    this._makeAttachments = function(doc, url, config, item) {
        if (config instanceof Array) {
            config.forEach(function (child) { this._makeAttachments(doc, url, child, item); }, this);
        } else if (typeof config === 'object') {
            /* plural or singual */
            var urlsFilter = config["urls"] || config["url"];
            var typesFilter = config["types"] || config["type"];
            var titlesFilter = config["titles"] || config["title"];
            var snapshotsFilter = config["snapshots"] || config["snapshot"];

            var attachUrls = this.evaluateThing(urlsFilter, doc, url);
            var attachTitles = this.evaluateThing(titlesFilter, doc, url);
            var attachTypes = this.evaluateThing(typesFilter, doc, url);
            var attachSnapshots = this.evaluateThing(snapshotsFilter, doc, url);

            if (!(attachUrls instanceof Array)) {
                attachUrls = [attachUrls];
            }
            for (var k in attachUrls) {
                var attachUrl = attachUrls[k];
                var attachType;
                var attachTitle;
                var attachSnapshot;
                if (attachTypes instanceof Array) { attachType = attachTypes[k]; }
                else { attachType = attachTypes; }

                if (attachTitles instanceof Array) { attachTitle = attachTitles[k]; }
                else { attachTitle = attachTitles; }

                if (attachSnapshots instanceof Array) { attachSnapshot = attachSnapshots[k]; }
                else { attachSnapshot = attachSnapshots; }

                item["attachments"].push({ url      : attachUrl,
                                           title    : attachTitle,
                                           mimeType : attachType,
                                           snapshot : attachSnapshot });
            }
        }
    };

    this.makeItems = function (doc, url, ignore, eachItem, ret) {
        var item = new Zotero.Item(this.itemType);
        item.url = url;
        for (var i in this._singleFieldNames) {
            var field = this._singleFieldNames[i];
            if (this[field]) {
                var fieldVal = this.evaluateThing(this[field], doc, url);
                if (fieldVal instanceof Array) {
                    item[field] = fieldVal[0];
                } else {
                    item[field] = fieldVal;
                }
            }
        }
        var multiFields = ["creators", "tags"];
        for (var j in multiFields) {
            var key = multiFields[j];
            var val = this.evaluateThing(this[key], doc, url);
            if (val) {
                for (var k in val) {
                    item[key].push(val[k]);
                }
            }
        }
        this._makeAttachments(doc, url, this["attachments"], item);
        eachItem(item, this, doc, url);
        ret();
    };
};

FW._Scraper.prototype = new FW._Base;

FW.MultiScraper = function (init) { 
    FW._scrapers.push(new FW._MultiScraper(init));
};

FW._MultiScraper = function (init) {
    for (x in init) {
        this[x] = init[x];
    }

    this._mkSelectItems = function(titles, urls) {
        var items = new Object;
        for (var i in titles) {
            items[urls[i]] = titles[i];
        }
        return items;
    };

    this._selectItems = function(titles, urls, callback) {
        var items = new Array();
	Zotero.selectItems(this._mkSelectItems(titles, urls), function (chosen) {
	    for (var j in chosen) {
		items.push(j);
	    }
	    callback(items);
	});
    };

    this._mkAttachments = function(doc, url, urls) {
        var attachmentsArray = this.evaluateThing(this['attachments'], doc, url);
        var attachmentsDict = new Object();
        if (attachmentsArray) {
            for (var i in urls) {
                attachmentsDict[urls[i]] = attachmentsArray[i];
            }
        }
        return attachmentsDict;
    };

    /* This logic is very similar to that used by _makeAttachments in
     * a normal scraper, but abstracting it out would not achieve much
     * and would complicate it. */
    this._makeChoices = function(config, doc, url, choiceTitles, choiceUrls) {
        if (config instanceof Array) {
            config.forEach(function (child) { this._makeTitlesUrls(child, doc, url, choiceTitles, choiceUrls); }, this);
        } else if (typeof config === 'object') {
            /* plural or singual */
            var urlsFilter = config["urls"] || config["url"];
            var titlesFilter = config["titles"] || config["title"];

            var urls = this.evaluateThing(urlsFilter, doc, url);
            var titles = this.evaluateThing(titlesFilter, doc, url);

            var titlesIsArray = (titles instanceof Array);
            if (!(urls instanceof Array)) {
                urls = [urls];
            }
            for (var k in urls) {
                var myUrl = urls[k];
                var myTitle;
                if (titlesIsArray) { myTitle = titles[k]; }
                else { myTitle = titles; }
                choiceUrls.push(myUrl);
                choiceTitles.push(myTitle);
            }
        }
    };

    this.makeItems = function(doc, url, ignore, eachItem, ret) {
        if (this.beforeFilter) {
            var newurl = this.beforeFilter(doc, url);
            if (newurl != url) {
                this.makeItems(doc, newurl, ignore, eachItem, ret);
                return;
            }
        }
        var titles = [];
        var urls = [];
        this._makeChoices(this["choices"], doc, url, titles, urls);
        var attachments = this._mkAttachments(doc, url, urls);
        
	var parentItemTrans = this.itemTrans;
	this._selectItems(titles, urls, function (itemsToUse) {
	    if(!itemsToUse) {
		ret();
	    } else {
	        var cb = function (doc1) {
		    var url1 = doc1.documentURI;
		    var itemTrans = parentItemTrans;
		    if (itemTrans === undefined) {
			itemTrans = FW.getScraper(doc1, url1);
		    }
		    if (itemTrans === undefined) {
			/* nothing to do */
		    } else {
			itemTrans.makeItems(doc1, url1, attachments[url1],
                                            eachItem, function() {});
		    }
		};
	        Zotero.Utilities.processDocuments(itemsToUse, cb, ret);
	    }
	});
    };
};

FW._MultiScraper.prototype = new FW._Base;

FW.WebDelegateTranslator = function (init) { 
    return new FW._WebDelegateTranslator(init);
};

FW._WebDelegateTranslator = function (init) {
    for (x in init) {
        this[x] = init[x];
    }
    this.makeItems = function(doc, url, attachments, eachItem, ret) {
        // need for scoping
        var parentThis = this;

        var translator = Zotero.loadTranslator("web");
        translator.setHandler("itemDone", function(obj, item) { 
            eachItem(item, parentThis, doc, url);
        });
        translator.setDocument(doc);

        if (this.translatorId) {
            translator.setTranslator(this.translatorId);
            translator.translate();
        } else {
            translator.setHandler("translators", function(obj, translators) {
                if (translators.length) {
                    translator.setTranslator(translators[0]);
                    translator.translate();
                }
            });
            translator.getTranslators();
        }
        ret();
    };
};

FW._WebDelegateTranslator.prototype = new FW._Base;

FW._StringMagic = function () {
    this._filters = new Array();

    this.addFilter = function(filter) {
        this._filters.push(filter);
        return this;
    };

    this.split = function(re) {
        return this.addFilter(function(s) {
            return s.split(re).filter(function(e) { return (e != ""); });
        });
    };

    this.replace = function(s1, s2, flags) {
        return this.addFilter(function(s) {
            if (s.match(s1)) {
                return s.replace(s1, s2, flags);
            } else {
                return s;
            }
        });
    };

    this.prepend = function(prefix) {
        return this.replace(/^/, prefix);
    };

    this.append = function(postfix) {
        return this.replace(/$/, postfix);
    };

    this.remove = function(toStrip, flags) {
        return this.replace(toStrip, '', flags);
    };

    this.trim = function() {
        return this.addFilter(function(s) { return Zotero.Utilities.trim(s); });
    };

    this.trimInternal = function() {
        return this.addFilter(function(s) { return Zotero.Utilities.trimInternal(s); });
    };

    this.match = function(re, group) {
        if (!group) group = 0;
        return this.addFilter(function(s) { 
                                  var m = s.match(re);
                                  if (m === undefined || m === null) { return undefined; }
                                  else { return m[group]; } 
                              });
    };

    this.cleanAuthor = function(type, useComma) {
        return this.addFilter(function(s) { return Zotero.Utilities.cleanAuthor(s, type, useComma); });
    };

    this.key = function(field) {
        return this.addFilter(function(n) { return n[field]; });
    };

    this.capitalizeTitle = function() {
        return this.addFilter(function(s) { return Zotero.Utilities.capitalizeTitle(s); });
    };

    this.unescapeHTML = function() {
        return this.addFilter(function(s) { return Zotero.Utilities.unescapeHTML(s); });
    };

    this.unescape = function() {
        return this.addFilter(function(s) { return unescape(s); });
    };

    this._applyFilters = function(a, doc1) {
        for (i in this._filters) {
            a = flatten(a);
            /* remove undefined or null array entries */
            a = a.filter(function(x) { return ((x !== undefined) && (x !== null)); });
            for (var j = 0 ; j < a.length ; j++) {
                try {
                    if ((a[j] === undefined) || (a[j] === null)) { continue; }
                    else { a[j] = this._filters[i](a[j], doc1); }
                } catch (x) {
                    a[j] = undefined;
                    Zotero.debug("Caught exception " + x + "on filter: " + this._filters[i]);
                }
            }
            /* remove undefined or null array entries */
            /* need this twice because they could have become undefined or null along the way */
            a = a.filter(function(x) { return ((x !== undefined) && (x !== null)); });
        }
        return flatten(a);
    };
};

FW.PageText = function () {
    return new FW._PageText();
};

FW._PageText = function() {
    this._filters = new Array();

    this.evaluate = function (doc) {        
        var a = [doc.documentElement.innerHTML];
        a = this._applyFilters(a, doc);
        if (a.length == 0) { return false; }
        else { return a; }
    };
};

FW._PageText.prototype = new FW._StringMagic();

FW.Url = function () { return new FW._Url(); };

FW._Url = function () {
    this._filters = new Array();

    this.evaluate = function (doc, url) {        
        var a = [url];
        a = this._applyFilters(a, doc);
        if (a.length == 0) { return false; }
        else { return a; }
    };
};

FW._Url.prototype = new FW._StringMagic();

FW.Xpath = function (xpathExpr) { return new FW._Xpath(xpathExpr); };

FW._Xpath = function (_xpath) {
    this._xpath = _xpath;
    this._filters = new Array();

    this.text = function() {
        var filter = function(n) {
            if (typeof n === 'object' && n.textContent) { return n.textContent; }
            else { return n; }
        };
        this.addFilter(filter);
        return this;
    };

    this.sub = function(xpath) {
        var filter = function(n, doc) {
            var result = doc.evaluate(xpath, n, null, XPathResult.ANY_TYPE, null);
            if (result) {
                return result.iterateNext();
            } else {
                return undefined;               
            }
        };
        this.addFilter(filter);
        return this;
    };

    this.evaluate = function (doc) {
        var res = doc.evaluate(this._xpath, doc, null, XPathResult.ANY_TYPE, null);
        var resultType = res.resultType;
        var a = new Array();
        if (resultType == XPathResult.STRING_TYPE) {
            a.push(res.stringValue);
        } else if (resultType == XPathResult.BOOLEAN_TYPE) {
            a.push(res.booleanValue);
        } else if (resultType == XPathResult.NUMBER_TYPE) {
            a.push(res.numberValue);
        } else if (resultType == XPathResult.ORDERED_NODE_ITERATOR_TYPE ||
                   resultType == XPathResult.UNORDERED_NODE_ITERATOR_TYPE) {
            var x;
            while ((x = res.iterateNext())) { a.push(x); }
        } 
        a = this._applyFilters(a, doc);
        if (a.length == 0) { return false; }
        else { return a; }
    };
};

FW._Xpath.prototype = new FW._StringMagic();

FW.detectWeb = function (doc, url) {
    for (var i in FW._scrapers) {
	var scraper = FW._scrapers[i];
	var itemType = scraper.evaluateThing(scraper['itemType'], doc, url);
	var v = scraper.evaluateThing(scraper['detect'], doc, url);
        if (v.length > 0 && v[0]) {
	    return itemType;
	}
    }
    return undefined;
};

FW.getScraper = function (doc, url) {
    var itemType = FW.detectWeb(doc, url);
    return FW._scrapers.filter(function(s) {
        return (s.evaluateThing(s['itemType'], doc, url) == itemType)
		&& (s.evaluateThing(s['detect'], doc, url));
    })[0];
};

FW.doWeb = function (doc, url) {
    var scraper = FW.getScraper(doc, url);
    scraper.makeItems(doc, url, [], 
                      function(item, scraper, doc, url) {
                          scraper.callHook('scraperDone', item, doc, url);
                          if (!item['title']) {
                              item['title'] = "";
                          }
                          item.complete();
                      },
                      function() {
                          Zotero.done();
                      });
    Zotero.wait();
};

/*********************** END FRAMEWORK ***********************/



/*
	***** BEGIN LICENSE BLOCK *****

	Kommersant Translator
	Copyright ?? 2011 Avram Lyon, ajlyon@gmail.com

	This file is part of Zotero.

	Zotero is free software: you can redistribute it and/or modify
	it under the terms of the GNU Affero General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	Zotero is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU Affero General Public License for more details.

	You should have received a copy of the GNU Affero General Public License
	along with Zotero. If not, see <http://www.gnu.org/licenses/>.

	***** END LICENSE BLOCK *****
*/



function detectWeb(doc, url) { 
		return FW.detectWeb(doc, url);
}
function doWeb(doc, url) { return FW.doWeb(doc, url); }

//technically these should be (and used to be) different item types and we should account for them, but this at least makes this work again in a basic fashion
/** Articles */
FW.Scraper({ itemType         : 'magazineArticle',
			 detect           : FW.Xpath('//h2[@class="article_name"]'),
			 title            : FW.Xpath('//h2[@class="article_name"]').text().trim(),
			 attachments      : [
			 { 
				 url : FW.Url(),
				 type : "text/html",
				 title : "Kommersant Snapshot"
		   } ],
			 creators         : FW.Xpath('//li[@rel="author"]/a[2]').text().split(",").cleanAuthor("author"),
			 date             : FW.Xpath('//div[contains(@class, "b-article_issue_number")]').text().match(/\d{2}\.\d{2}\.\d{4}/),
			 issue            : FW.Xpath('//div[contains(@class, "b-article_issue_number")]/a').text().match(/\d+/),
			 abstractNote     : FW.Xpath('//span[@class="b-article__intro"]').text().trimInternal(),
			 url			  : FW.Url().replace(/(\?|&)stamp.+/, ""),
			 pages			  : FW.Xpath('//div[contains(@class, "b-article_issue_number")]').text().match(/??????.\s*\d+/).remove(/??????.\s*/),
			 publicationTitle : FW.Xpath('//div[contains(@class, "b-article_issue_number")]').text().match(/.+???/).remove(/???/).remove(/\. ????????????????????/),
			 hooks : { "scraperDone": function  (item,doc, url) {
				if (!item.publicationTitle) item.publicationTitle = "??????????????????????";	
			 }}
});

/** Search results */
FW.MultiScraper({ itemType  : "multiple",
				  detect    : FW.Xpath('//div[contains(@class,"search-results_list")]'),
				  choices : {
					titles    : FW.Xpath('//h4[@class="article_name"]/a').text(),
				  	urls      : FW.Xpath('//h4[@class="article_name"]/a').key('href').text()         
				  }
});

/** BEGIN TEST CASES **/
var testCases = [
	{
		"type": "web",
		"url": "http://kommersant.ru/doc/1811182",
		"items": [
			{
				"itemType": "magazineArticle",
				"title": "?? ???????????? ???????????????? ???? ?????????????? ????????????????????????",
				"creators": [
					{
						"firstName": "??????????????",
						"lastName": "????????????",
						"creatorType": "author"
					}
				],
				"date": "08.11.2011",
				"abstractNote": "?????? ?????????? ???????????????? \"??\", ?????????? ???????????????????????? ???????????????? ?????????????????????? ?????????? (??????) ???????????? ?????????????? ?? ???????? 2012 ???????? ?????????? ?????????????? ????????????: ???? ???? ?????????????? ?????????????????????? ?????????? ?? ?????????????????????? ?????????????????????????? ???????????????? (??????) ???? ?????????????? ????. ?????????????????????? ???????????????????? ?????? ?????????? ?????????????????? ???????????????????????????? ????????????????, ?????????? ?????????????? ?? ?????????????????????? ?????????????????? ???????????? ??????????. ?????????????? ?????????????????? ?? ?????????? ???????????????????????? ?????? ??? \"?????????????? ?????????????????? ?? ???????????????? ?? ????????????????????????????????????\": ???? ?? ???????????????????? ?????????????? ?????????? ???????????????????? ?????????????????? ?? ?????? ?????????????????????? ???????????????????? ??????????????????????.",
				"issue": "208",
				"libraryCatalog": "Kommersant",
				"pages": "1",
				"publicationTitle": "???????????? \"??????????????????????\"",
				"url": "http://kommersant.ru/doc/1811182",
				"attachments": [
					{
						"title": "Kommersant Snapshot",
						"mimeType": "text/html"
					}
				],
				"tags": [],
				"notes": [],
				"seeAlso": []
			}
		]
	},
	{
		"type": "web",
		"url": "http://www.kommersant.ru/doc/1832739?stamp=634721007709478300",
		"items": [
			{
				"itemType": "magazineArticle",
				"title": "???????????????? ??????",
				"creators": [],
				"date": "12.12.2011",
				"abstractNote": "???? ??????, ?????? ???????????????? ???????????????????? ???????????? ?? ??????????, ?????? ???????????????????????????????? ???????????? ???? ???????????????????????????? ???????????????? ????????????????, ???????????????? ?????????????????????????? \"????????????\" ?????????? ????????????.",
				"issue": "49",
				"libraryCatalog": "Kommersant",
				"pages": "28",
				"publicationTitle": "???????????? \"?????????????????????? ????????????\"",
				"url": "http://www.kommersant.ru/doc/1832739",
				"attachments": [
					{
						"title": "Kommersant Snapshot",
						"mimeType": "text/html"
					}
				],
				"tags": [],
				"notes": [],
				"seeAlso": []
			}
		]
	},
	{
		"type": "web",
		"url": "http://www.kommersant.ru/doc/1836636?themeid=589",
		"items": [
			{
				"itemType": "magazineArticle",
				"title": "\"?????? ???? ???????????????????????? ??????????????????????, ???? ???? ???????? ????????????, ?? ???? ????????????, ???? ????????????????????\"",
				"creators": [],
				"date": "12.12.2011",
				"libraryCatalog": "Kommersant",
				"publicationTitle": "??????????????????????",
				"url": "http://www.kommersant.ru/doc/1836636?themeid=589",
				"attachments": [
					{
						"title": "Kommersant Snapshot",
						"mimeType": "text/html"
					}
				],
				"tags": [],
				"notes": [],
				"seeAlso": []
			}
		]
	},
	{
		"type": "web",
		"url": "http://kommersant.ru/Search/Results?places=1%2C6%2C34%2C52%2C61%2C62%2C198%2C2%2C3%2C84%2C17%2C14%2C5%2C217%2C66%2C57%2C210%2C86%2C9999&categories=20&isbankrupt=false&datestart=8.10.2011&dateend=8.11.2011&sort_type=0&sort_dir=0&region_selected=-1&results_count=300&saved_query=&saved_statement=&page=1&search_query=%CF%F3%D2%E8%CD&stamp=634721009393393053",
		"defer": true,
		"items": "multiple"
	}
]
/** END TEST CASES **/