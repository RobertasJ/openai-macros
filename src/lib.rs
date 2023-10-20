use proc_macro::TokenStream;
use syn::spanned::Spanned;
use proc_macro2::Ident;
use quote::{quote, ToTokens};
use syn::parse::{Parse, ParseStream};
use syn::{Data, DeriveInput, Fields, FieldValue, Type};
use syn::{parse, parse_macro_input, Expr, Member, Token};
use syn::punctuated::Punctuated;
use syn::token::Colon;

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
                let f_val: proc_macro::TokenStream = field.expr.clone().into_token_stream().into();
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
                let f_val: proc_macro::TokenStream = field.expr.clone().into_token_stream().into();
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

struct FromAgents {
    agent_field: AgentField,
    _semicolon_token1: Token![;],
    input_type: Type,
    _semicolon_token2: Token![;],
    output_type: Type,
    _semicolon_token3: Token![;],
    agents: Punctuated<Type, Token![,]>,
}

struct AgentField {
    ident: Type,
    _semi_token: Option<Token![:]>,
    semi_type: Option<Type>
}

impl Parse for AgentField {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident = input.parse()?;
        let semi_token: Option<Colon> = input.parse()?;
        let mut semi_type = None;
        if semi_token.is_some() {
            semi_type = Some(input.parse()?);
        }

        Ok(Self {
            _semi_token: semi_token,
            ident,
            semi_type,
        })
    }
}

impl Parse for FromAgents {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            agent_field: input.parse()?,
            _semicolon_token1: input.parse()?,
            input_type: input.parse()?,
            _semicolon_token2: input.parse()?,
            output_type: input.parse()?,
            _semicolon_token3: input.parse()?,
            agents: input.parse_terminated(Type::parse, Token![,])?,
        })
    }
}


/// this is a static way to make a chain of agents
#[proc_macro]
pub fn from_agents(input: TokenStream) -> TokenStream {
    let FromAgents { agents, input_type, output_type, agent_field: agent_ident, .. } = parse_macro_input!(input as FromAgents);

    let (name, memoized) = {
        let AgentField { ident, semi_type, .. } = agent_ident;
        let memoized = if let Some(t) = semi_type {
            if t.to_token_stream().to_string() == "Memoized" {
                true
            } else { false }
        } else { false };

        (ident, memoized)
    };

    let o = match memoized {
        true => quote! {
            #[derive(openai_macros::Memoize, Default)]
            struct #name {
                memo: Option<#output_type>,
            }

            impl openai_utils::MemoizedAgent for #name {
                type Input = #input_type;

                async fn computation(&mut self, input: Self::Input) -> Self::Output {
                    use openai_macros::evaluate_chain;
                    evaluate_chain!(input; #agents)
                }
            }
        },
        false => quote! {
            #[derive(Default)]
            struct #name;

            impl openai_utils::Agent for #name {
                type Input = #input_type;
                type Output = #output_type;

                async fn compute(&mut self, input: Self::Input) -> Self::Output {
                    use openai_macros::evaluate_chain;
                    evaluate_chain!(input; #agents)
                }
            }
        }
    };


    o.into()

}

struct AgentChain {
    input: Expr,
    _semicolon_token1: Token![;],
    agents: Punctuated<Type, Token![,]>,
}

impl Parse for AgentChain {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(Self {
            input: input.parse()?,
            _semicolon_token1: input.parse()?,
            agents: input.parse_terminated(Type::parse, Token![,])?,
        })
    }
}

#[proc_macro]
pub fn evaluate_chain(input: TokenStream) -> TokenStream {
    let AgentChain { agents, input, .. } = parse_macro_input!(input as AgentChain);

    let agent_chain = agents.iter().fold(quote! { #input }, |acc, a| {
        quote! {
            #a::default().compute(#acc).await
        }
    });

    agent_chain.into()
}

#[proc_macro_derive(Memoize, attributes(memo))]
pub fn memoized(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let Data::Struct(s) = &input.data else {
        return syn::Error::new(input.ident.span(), "not a struct").to_compile_error().into();
    };
    let Fields::Named(fields) = &s.fields else {
        return syn::Error::new(s.fields.span(), "not a named fields struct").to_compile_error().into();
    };

    let field = fields.named.iter().find(|f| {
        f.attrs.iter().any(|attr| attr.path().is_ident("memo"))
    });

    let (ident, ty) = match field {
        Some(field) => {
            // Found the field with the `#[memo]` attribute.
            // Use this field.
            (field.ident.clone(), field.ty.clone())
        },
        None => {
            // No field with the `#[memo]` attribute was found.
            // Default to the `memo` field.
            let default_field = fields.named.iter().find(|f| f.ident.as_ref().map(|i| i == "memo").unwrap_or(false));
            match default_field {
                Some(field) => (field.ident.clone(), field.ty.clone()),
                None => panic!("No field named 'memo' found in the struct."),
            }
        }
    };

    let struct_name = input.ident;
    let ty = match &ty {
        Type::Path(type_path) => {
            // Check if the field type is Option<T>
            if let Some(segment) = type_path.path.segments.last() {
                if segment.ident == "Option" {
                    // Extract the inner type T
                    if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
                        if let syn::GenericArgument::Type(inner_ty) = args.args.first().unwrap() {
                            inner_ty.clone()
                        } else {
                            panic!("Invalid field type for memoization. Expected Option<TheTypeWeWant>.")
                        }
                    } else {
                        panic!("Invalid field type for memoization. Expected Option<TheTypeWeWant>.")
                    }
                } else {
                    panic!("Invalid field type for memoization. Expected Option<TheTypeWeWant>.")
                }
            } else {
                panic!("Invalid field type for memoization. Expected Option<TheTypeWeWant>.")
            }
        },
        _ => panic!("Invalid field type for memoization. Expected Option<TheTypeWeWant>.")
    };
    let output = quote! {
        impl openai_utils::Memoized for #struct_name {
            type Output = #ty;

            fn field(&mut self) -> Option<&mut Self::Output> {
                self. #ident .as_mut()
            }
        }

    };

    output.into()
}