// address of the notebook
var address = "http://localhost:8888/notebooks/phantomjs-tryout.ipynb"
// auth token from Jupyter console
var authToken = "b05a65d622136665a1292b9f3a29192364eaccb4bfbc6887"
// cell number with a widget output
var cellNumber = 8;

// this function is used to verify that a page is fully loaded
// source: https://github.com/ariya/phantomjs/blob/master/examples/waitfor.js
function waitFor(testFx, onReady, timeOutMillis) {
    var maxtimeOutMillis = timeOutMillis ? timeOutMillis : 3000,
        start = new Date().getTime(),
        condition = false,
        interval = setInterval(function() {
            if ( (new Date().getTime() - start < maxtimeOutMillis) && !condition ) {
                // If not time-out yet and condition not yet fulfilled
                condition = (typeof(testFx) === "string" ? eval(testFx) : testFx());
            } else {
                if(!condition) {
                    // If condition still not fulfilled (timeout but condition is 'false')
                    console.log("'waitFor()' timeout");
                    phantom.exit(1);
                } else {
                    // Condition fulfilled (timeout and/or condition is 'true')
                    console.log("'waitFor()' finished in " + (new Date().getTime() - start) + "ms.");
                    typeof(onReady) === "string" ? eval(onReady) : onReady();
                    clearInterval(interval); //< Stop this interval
                }
            }
        }, 250); //< repeat check every 250ms
};

// log in to a notebook using a token
function logIn() {
    console.log("Logging in");
    page.evaluate(function(token) {
        document.forms[0].password.value = token;
        document.forms[0].submit();
    }, authToken);
}

// wait for a notebook to fully load, find the
// needed output cell and save it as a PNG file
function saveAsPNG() {
    console.log("Saving PNG")
    // Wait for 'notebook-container' to be visible
    waitFor(function() {
        // Check in the page if a specific element is now visible
        return page.evaluate(function() {
            return $("#notebook-container").is(":visible");
        });
    }, function() {
        console.log("The notebook-container element should be visible now.");
        var clipRect = page.evaluate(function(cell){
            // we are selecting only the output cell
            var searchStr = 'div.input_prompt:contains("[' + cell + ']:")';
            var parentCell = $(searchStr).parents('div.cell')[0];
            // get only the data div
            var outputSubarea = $(parentCell).find('div.output_subarea')[0];
            // get the coordinates of the data div
            return outputSubarea.getBoundingClientRect()
        }, cellNumber);

        page.clipRect = {
            top:    clipRect.top,
            left:   clipRect.left,
            width:  clipRect.width,
            height: clipRect.height
          };
       page.render('example.png');
       phantom.exit();
    });
}

var page = require('webpage').create();
// it seems, viewportSize should fully cover the
// the rendered div position, or nothing will be saved.
page.viewportSize = { width: 5000, height: 5000 };

page.open(address, function (status) {
    // Check for page load success
    if (status !== "success") {
        console.log("Unable to open a page");
    } else {
        // Wait for 'password_input' to be visible
        waitFor(function() {
            // Check in the page if a specific element is now visible
            return page.evaluate(function() {
                return $("#password_input").is(":visible");
            });
        }, function() {
           console.log("The password_input element should be visible now.");
           logIn();
           saveAsPNG();
        });
    }
});

