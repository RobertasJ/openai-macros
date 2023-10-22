use proc_macro::TokenStream;
use proc_macro2::Ident;
use quote::{quote, ToTokens};
use syn::parse::{Parse, ParseStream};
use syn::FieldValue;
use syn::{parse, parse_macro_input, Expr, Member, Token};

enum MessageType {
    Assistant,
    User,
    Function,
    System,
}

impl Parse for MessageType {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        Ok(match ident.to_string().as_str() {
            "user" => MessageType::User,
            "assistant" => MessageType::Assistant,
            "function" => MessageType::Function,
            "system" => MessageType::System,
            _ => return Err(input.error("unexpected message type")),
        })
    }
}

struct Message {
    message_type: MessageType,
    user_name: Option<Expr>,
    content: Option<Expr>,
}

impl Parse for Message {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        macro_rules! find_pred {
            ($l:expr) => {
                |f: &&FieldValue| -> bool {
                    if let Member::Named(name) = &f.member {
                        name.to_string().as_str() == $l
                    } else {
                        false
                    }
                }
            };
        }

        let fields = input.parse_terminated(FieldValue::parse, Token![,])?;

        Ok(Message {
            message_type: {
                let fields = fields
                    .iter()
                    .filter(|f| f.colon_token.is_none())
                    .cloned()
                    .collect::<Vec<FieldValue>>();
                let field = if fields.len() == 1 {
                    fields[0].clone()
                } else {
                    return Err(input.error("message type not specified"));
                };
                let f_val: TokenStream = field.expr.clone().into_token_stream().into();
                let m_type = parse(f_val)?;
                m_type
            },
            user_name: {
                let field = fields.iter().find(find_pred!("user_name"));
                if let Some(f) = field {
                    Some(f.expr.clone())
                } else {
                    None
                }
            },
            content: {
                let field = fields.iter().find(find_pred!("content"));
                if let Some(f) = field {
                    Some(f.expr.clone())
                } else {
                    None
                }
            },
        })
    }
}

#[proc_macro]
pub fn message(input: TokenStream) -> TokenStream {
    let message = parse_macro_input!(input as Message);

    let m_type = match message.message_type {
        MessageType::Assistant => "assistant",
        MessageType::User => "user",
        MessageType::Function => "function",
        MessageType::System => "system",
    };

    let mut output = quote! {
            use openai_utils::Message;
            Message::new(#m_type)
    };

    if let Some(content) = message.content {
        output = quote! {
            #output.with_content(#content)
        };
    }

    if let Some(user) = message.user_name {
        output = quote! {
            #output.with_user(#user)
        };
    }

    output = quote! {
        {
            #output
        }
    };

    output.into()
}

struct AiAgent {
    model: Expr,
    messages: Option<Expr>,
    function_call: Option<Expr>,
    temperature: Option<Expr>,
    top_p: Option<Expr>,
    n: Option<Expr>,
    stop: Option<Expr>,
    max_tokens: Option<Expr>,
    presence_penalty: Option<Expr>,
    frequency_penalty: Option<Expr>,
    system_message: Option<Expr>,
    logit_bias: Option<Expr>,
    user: Option<Expr>,
}

impl Parse for AiAgent {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        macro_rules! find_pred {
            ($l:expr) => {
                |f: &&FieldValue| -> bool {
                    if let Member::Named(name) = &f.member {
                        name.to_string().as_str() == $l
                    } else {
                        false
                    }
                }
            };
        }

        let fields = input.parse_terminated(FieldValue::parse, Token![,])?;

        let required_field = |l: &str| {
            let fields = fields
                .iter()
                .filter(find_pred!(l))
                .cloned()
                .collect::<Vec<FieldValue>>();
            if fields.len() == 1 {
                let field = fields[0].clone();
                let f_val: TokenStream = field.expr.clone().into_token_stream().into();
                parse(f_val)
            } else {
                Err(input.error(format!("'{}' field not specified", l)))
            }
        };

        let optional_field = |l: &str| {
            let field = fields.iter().find(find_pred!(l));
            if let Some(f) = field {
                Some(f.expr.clone())
            } else {
                None
            }
        };

        Ok(Self {
            model: required_field("model")?,
            messages: optional_field("messages"),
            function_call: optional_field("function_call"),
            temperature: optional_field("temperature"),
            top_p: optional_field("top_p"),
            n: optional_field("n"),
            stop: optional_field("stop"),
            max_tokens: optional_field("max_tokens"),
            presence_penalty: optional_field("presence_penalty"),
            frequency_penalty: optional_field("frequency_penalty"),
            system_message: optional_field("system_message"),
            logit_bias: optional_field("logit_bias"),
            user: optional_field("user"),
        })
    }
}

#[proc_macro]
pub fn ai_agent(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as AiAgent);

    let model = input.model;

    let mut b = quote! {
        openai_utils::AiAgent::new(#model)
    };

    if let Some(messages) = input.messages {
        if let Expr::Array(messages) = messages {
            b = quote! {
                #b.with_messages(#messages.into())
            };
        } else {
            b = quote! {
                #b.with_messages([#messages].into())
            };
        }


    } else {
        b = quote! {
            #b.with_messages(vec![])
        };
    }

    if let Some(function_call) = input.function_call {
        b = quote! {
            #b.with_function_call(#function_call)
        };
    }

    if let Some(temperature) = input.temperature {
        b = quote! {
            #b.with_temperature(#temperature)
        };
    }

    if let Some(top_p) = input.top_p {
        b = quote! {
            #b.with_top_p(#top_p)
        };
    }

    if let Some(n) = input.n {
        b = quote! {
            #b.with_n(#n)
        };
    }

    if let Some(stop) = input.stop {
        b = quote! {
            #b.with_stop(#stop)
        };
    }

    if let Some(max_tokens) = input.max_tokens {
        b = quote! {
            #b.with_max_tokens(#max_tokens)
        };
    }

    if let Some(presence_penalty) = input.presence_penalty {
        b = quote! {
            #b.with_presence_penalty(#presence_penalty)
        };
    }

    if let Some(frequency_penalty) = input.frequency_penalty {
        b = quote! {
            #b.with_frequency_penalty(#frequency_penalty)
        };
    }

    if let Some(logit_bias) = input.logit_bias {
        b = quote! {
            #b.with_logit_bias(#logit_bias)
        };
    }

    if let Some(system_message) = input.system_message {
        b = quote! {
            #b.with_system_message(#system_message)
        };
    }

    if let Some(user) = input.user {
        b = quote! {
            #b.with_user(#user)
        };
    }


    b.into()
}