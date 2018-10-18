window.onload = function() {
    authenticate();
    postcodeInput = document.getElementById('postcodeInput'); //postcode input element
}

var url = "http://127.0.0.1:5100";
var apiToken = null;
var authCredentials = {
    'username' : 'Tony',
    'passowrd' : '123456'
}

function authenticate() {
    $.getJSON(url + `/token`, authCredentials, (data) => {
        apiToken = data['token']
        // console.log(apiToken)
    })
}

function setHeader(xhr) {
    xhr.setRequestHeader('AUTH-TOKEN', apiToken);
}

// used in brandDoneTyping to only make an API call if brand exists
// within list
var savedBrandList = [];

// user is "finished typing" postcode, update Brand list
function postcodeDoneTyping (){
    list = document.getElementById("Fuel Brand");
    // console.log(document.getElementById("Fuel Brand"));

    let postcode = document.getElementById('postcodeInput').value;
    let args = { "postcode" : postcode };
    if (postcode.match(/\d{4}/)) {
        $.ajax({
            url: url + '/getBrands',
            type: 'GET',
            dataType: 'json',
            data: args,
            success: function(data) {
                clearBrandList();
                clearTypeList();
                data.forEach(element => {
                    // console.log(element);
                    var opt = document.createElement("option")
                    opt.value = element;
                    list.appendChild(opt);
                    savedBrandList.push(element)
                });
            },
            error: function() {
                console.log("Oops");
            },
            beforeSend: setHeader
        });
    }
}

// user is "finished typing" brand, update Type list
function brandDoneTyping() {
    list = document.getElementById("Fuel Type");
    // console.log(document.getElementById("Fuel Type"));

    let postcode = document.getElementById('postcodeInput').value;
    let brand = document.getElementById('brandInput').value;
    let args = { "postcode" : postcode, "brand" : brand };
    if (postcode.match(/\d{4}/) && savedBrandList.includes(brand)) {
        $.ajax({
            url: url + '/getFuelTypes',
            type: 'GET',
            dataType: 'json',
            data: args,
            success: function(data) {
                clearTypeList();
                data.forEach(element => {
                    // console.log(element);
                    var opt = document.createElement("option")
                    opt.value = element;
                    list.appendChild(opt);
                });
            },
            error: function() {
                console.log("Oops");
            },
            beforeSend: setHeader
        });
    }
}

function clearBrandList() {
    brandList = document.getElementById("Fuel Brand");
    while (brandList.firstChild) {
        brandList.removeChild(brandList.firstChild);
    }
    savedBrandList = [];
}

function clearTypeList() {
    typeList = document.getElementById("Fuel Type");
    while (typeList.firstChild) {
        typeList.removeChild(typeList.firstChild);
    }
}
