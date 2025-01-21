
function Message() {
    
    const name = "Fang";
    if (name) {
        return <h1>Hey, {name}</h1>;
    }
    return<h1>Hello World</h1>;
}

export default Message;