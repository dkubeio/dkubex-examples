import { OPENAI_API_HOST, OPENAI_API_TYPE, OPENAI_API_VERSION, OPENAI_ORGANIZATION } from '@/utils/app/const';

import { OpenAIModel, OpenAIModelID, OpenAIModels } from '@/types/openai';

export const config = {
    runtime: 'edge',
};

const handler = async (req: Request): Promise<Response> => {
    try {
        const { key, request_id, feedback, message } = (await req.json()) as {
            request_id: string;
            feedback: string;
            message: string;
            key: string;
        };

        let url = `${OPENAI_API_HOST}/sgpt/feedback`;


        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...(OPENAI_API_TYPE === 'openai' && {
                    Authorization: `Bearer ${key ? key : process.env.OPENAI_API_KEY}`
                }),
                ...(OPENAI_API_TYPE === 'azure' && {
                    'api-key': `${key ? key : process.env.OPENAI_API_KEY}`
                }),
                ...((OPENAI_API_TYPE === 'openai' && OPENAI_ORGANIZATION) && {
                    'OpenAI-Organization': OPENAI_ORGANIZATION,
                }),
            },
            method: 'POST',
            body: JSON.stringify({
                request_id: request_id,
                feedback: feedback,
                message: message,
                // key: key
            }),
        });

        if (response.status === 401) {
            return new Response(response.body, {
                status: 500,
                headers: response.headers,
            });
        } else if (response.status !== 200) {
            console.error(
                `OpenAI API returned an error ${response.status
                }: ${await response.text()}`,
            );
            throw new Error('OpenAI API returned an error');
        }
        const json = await response.json();

        return new Response(JSON.stringify(json), { status: 200 });
    } catch (error) {
        console.error(error);
        return new Response('Error', { status: 500 });
    }
};

export default handler;
