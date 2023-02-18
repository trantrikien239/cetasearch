var FORM_ID_INCR = 0; // Ever-increasing couter
var array_name = [];

window.onload = function() {
    createAndAppendNewContactForm();
}

document.getElementById('add-form-btn').addEventListener('click', function(e) {
    createAndAppendNewContactForm();
});

function createAndAppendNewContactForm() {

    
    if (FORM_ID_INCR < 11) {
        //console.log(FORM_ID_INCR)
        let viewModel = { formId : FORM_ID_INCR};
        let template = document.getElementById('form-template').innerHTML;
        template.display = 'block';
        let renderedHtml = Mustache.render(template, viewModel);
        let node = document.createRange().createContextualFragment(renderedHtml);
        document.getElementById('form-container').appendChild(node);
        document.getElementById('form-container').children[FORM_ID_INCR].id = "form-"+FORM_ID_INCR
        // document.getElementById('form-container').children[FORM_ID_INCR].children[0].innerText = "Form " + (FORM_ID_INCR + 1)
    
        sp = document.getElementById("submit");
        sp.setAttribute("class","button");
        sp.removeAttribute("hidden");

        // sp = document.getElementById("visualization");
        // sp.setAttribute("class","button");
        // sp.removeAttribute("hidden");

        FORM_ID_INCR ++
        
    } else {
        alert('We only accept 10 people ratings at the same time');
    }
}

function func() {
    data = []
    console.log("Hahaha")
    for (var i = 0; i < FORM_ID_INCR; i++) {
        form = document.getElementById("form-"+i);
        console.log(form.querySelector("#name_input"));
        var username = form.querySelector("#name_input").value;
        form.querySelector("#name_input").value = "";

        var anime_names = [];
        var ratings = [];

        for (var j = 0; j <= 9; j++) {
            anime = form.querySelector("#anime_input_" + j).value;
            form.querySelector("#anime_input_" + j).value = "";
            rating = form.querySelector("#rating_input_" + j).value;
            form.querySelector("#rating_input_" + j).value = ""
    
            if (anime !=  "" && rating != "" && username != "" && !isNaN(rating)) {
                anime_names.push(anime);
                ratings.push(+rating);
            }
        }

        form_data = {
            "name": username,
            "animes": anime_names,
            "ratings": ratings
        }

        data.push(form_data)
        
    }

    console.log(data)

    $.post("/collect", {
        jdata: JSON.stringify(data)
    }).done(function(d) {
        sp = document.getElementById("visualization");
        sp.setAttribute("class","button");
        sp.removeAttribute("hidden");
    });
    
}